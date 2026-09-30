import os
import time
import torch
import transformers
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import threading
import asyncio
from src.cfg.constants import *

app = FastAPI(title="LLM API", version="1.0")

BATCH_SIZE = 8  # num of LLM requests to process at once
BATCH_WAIT_TIME = 2  # max wait time for batch to fill in s

class LLMRequest(BaseModel):
    prompt: str
    max_new_tokens: int = LLM_MAX_NEW_TOKENS
    top_p: float = 0.8
    temperature: float = 0.7

class LLMModel:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    print(f"Loading model at {MODEL_PATH} for the first time")
                    cls._instance = super(LLMModel, cls).__new__(cls)
                    cls._instance._initialize()
                    print('I created my instance')
                    print(dir(cls._instance))            
        return cls._instance
    
    def _initialize(self):
        print("initializing", flush=True)
        self._require_cuda()

        # TODO figure out how to better handle the initialization (i.e. mixtral dies because it doesn't have attention)
        # TODO find out why when this dies the code around it continues i.e. a model is returned to generate_text, but I never see the print out of "I created my instance"
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa" # faster inference
        ).eval()
        self._assert_model_on_gpu()
        print("model loaded", flush=True)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
        print("tokenizer created", flush=True)
        
        # decoder-only models need left padding for correct generation order
        self.tokenizer.padding_side = "left"

        # for batching, need to set pad tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Model already has device_map; do not pass device= to pipeline (it would
        # force CPU and print the misleading "Device set to use cpu" message).
        self.pipeline = transformers.pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            task="text-generation",
            temperature=0.1,
            top_p=0.15,
            top_k=0,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            repetition_penalty=1.1,
            do_sample=True,
            batch_size=BATCH_SIZE # for batch support
        )
        print("pipeline created", flush=True)
        
        self.request_queue = asyncio.Queue() # queue for holding requests to process
        self.batch_task = None # current task
        self.batch_lock = asyncio.Lock() # lock for
        self.is_processing = False # current state
        print("ready to go", flush=True)

    @staticmethod
    def _require_cuda():
        """Fail fast if GPUs are not visible — CPU 70B silently ruins islands runs."""
        print(
            f"torch={torch.__version__} cuda_build={torch.version.cuda} "
            f"cuda_available={torch.cuda.is_available()} "
            f"device_count={torch.cuda.device_count()} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
            flush=True,
        )
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError(
                "CUDA is not available to PyTorch. Refusing to load Llama on CPU. "
                "Check module load cuda, GPU allocation (gres), and the torch CUDA build."
            )
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(
                f"  GPU {i}: {props.name} "
                f"({props.total_memory / (1024**3):.1f} GiB)",
                flush=True,
            )

    def _assert_model_on_gpu(self):
        """Ensure device_map=auto did not fall back to an all-CPU placement."""
        device_map = getattr(self.model, "hf_device_map", None)
        print(f"hf_device_map={device_map}", flush=True)
        if device_map is not None:
            devices = {str(v) for v in device_map.values()}
            if devices and devices.issubset({"cpu", "disk"}):
                raise RuntimeError(
                    f"Model landed entirely on CPU/disk ({device_map}). "
                    "Aborting so islands do not run multi-hour CPU inference."
                )
            return
        param_devices = {str(p.device) for p in self.model.parameters()}
        print(f"param_devices={param_devices}", flush=True)
        if param_devices == {"cpu"}:
            raise RuntimeError(
                "Model parameters are all on CPU. Aborting GPU-required load."
            )
    
    async def start_batch_processor(self):
        """Start the batch processor if it's not already running"""
        async with self.batch_lock:
            if not self.is_processing:
                self.is_processing = True
                self.batch_task = asyncio.create_task(self._batch_processor())
    
    async def _batch_processor(self):
        """Process requests in batches"""
        try:
            while True:
                batch = []
                futures = []
                
                try:
                    # check queue for requests
                    request, future = await self.request_queue.get()
                    batch.append(request)
                    futures.append(future)
                    
                    # try to fill the batch until BATCH_SIZE or timeout
                    batch_start_time = time.time()
                    while len(batch) < BATCH_SIZE and (time.time() - batch_start_time) < BATCH_WAIT_TIME:
                        try:
                            req, fut = await asyncio.wait_for(
                                self.request_queue.get(),
                                timeout=max(0, BATCH_WAIT_TIME - (time.time() - batch_start_time))
                            )
                            batch.append(req)
                            futures.append(fut)
                        except asyncio.TimeoutError:
                            break
                    
                    batch_size = len(batch)
                    print(f"Processing batch of {batch_size} requests", flush=True)
                    
                    prompts = [req["prompt"] for req in batch]
                    
                    max_new_tokens = max(req["max_new_tokens"] for req in batch)
                    
                    # all temps and top_p are same
                    temperature = batch[0]["temperature"] 
                    top_p = batch[0]["top_p"]
                    
                    start_time = time.time()
                    
                    # Run blocking GPU inference off the asyncio event loop.
                    # Calling pipeline() directly here freezes uvicorn for minutes
                    # (no health checks, no new accepts) and can wedge the server
                    # under a flood of mutation clients.
                    results = await asyncio.to_thread(
                        self.pipeline,
                        prompts,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    
                    response_time = round(time.time() - start_time, 2)
                    print(f"Batch of {batch_size} finished in {response_time}s", flush=True)
                    
                    # for every future, set its result
                    for result, future in zip(results, futures):
                        output_txt = result[0].get("generated_text", str(result))
                        
                        future.set_result({
                            "generated_text": output_txt,
                            "response_time_sec": response_time,
                            "batch_size": batch_size
                        })
                        
                        # done with task
                        self.request_queue.task_done()
                
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                    for future in futures:
                        if not future.done():
                            future.set_exception(e)
                    
                    # Mark all tasks as done
                    for _ in range(len(futures)):
                        self.request_queue.task_done()
                
                # Check if queue is empty, if so - sleep briefly to save resources
                if self.request_queue.empty():
                    await asyncio.sleep(0.01)
        
        except asyncio.CancelledError:
            print("Batch processor cancelled")
        except Exception as e:
            print(f"Unexpected error in batch processor: {str(e)}")
        finally:
            async with self.batch_lock:
                self.is_processing = False
    
    async def generate(self, request_dict):
        """Submit a request to the batch processor"""
        # future is a placeholder for later result
        future = asyncio.Future()
        
        # put in queue
        print('Hey I am about to access the request queue attribute')
        await self.request_queue.put((request_dict, future))
        print('No problem, I got it')
        
        # start processing batches if not already started
        await self.start_batch_processor()
        
        # wait & return future result
        return await future

@app.post("/generate")
async def generate_text(request: LLMRequest):
    """
    Submits LLMRequest to the local model. If the model has not been intialized before, this function will first have to intialize the model.

    Parameters:
    LLMRequest:
        txt2llm (str): input to llm
        max_new_tokens (int): maximum number of tokens model should generate
        top_p (int): threshold, higher to consider wider range of words
        temperature (int): randomness, higher for more varied outputs

    Returns:
    dict: generated_text (output of LLM) and response_time (time after intializing model until generated text obtained)
    """
    try:
        # Convert request to dict
        request_dict = {
            "prompt": request.prompt,
            "max_new_tokens": request.max_new_tokens,
            "top_p": request.top_p,
            "temperature": request.temperature
        }
        
        # Get the model instance
        model = LLMModel()
        print(dir(model))
        
        # Submit to the batch processor and wait for result
        start_time = time.time()
        print(f"Request received at {time.strftime('%H:%M:%S', time.localtime(start_time))}")
        
        result = await model.generate(request_dict)
        
        print(f"Request completed in {time.time() - start_time:.2f}s")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "LLM API is running!"}

print('Server running with server-side batching!')


@app.on_event("startup")
async def preload_model():
    """Load the model at process start so first request is fast."""
    model = LLMModel()
    await model.start_batch_processor()
