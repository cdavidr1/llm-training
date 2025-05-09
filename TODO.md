[unsloth-finetuning]
>> [Dataset]
* >> Compatibility
  - * Alpaca = [{"ins":"","inp":"","output":""},{...}]
  - * Supervised instruction finetuning combines each training data into one prompt
  - * to_sharegpt :: May need to merge all columns into 1 prompt: unsloth uses to_sharegpt to perform this merge
  - * to_sharegpt :: conversation_extension creates a mixed conversation mode to train with conversational flow
  - * standardize_sharegpt :: formats correctly for finetuning which is called right after to_sharegpt


* Look into other techniques. LoRA trained finally but didnt yield good results
* possibly embeddings?

* RAG = Put knowledge into the prompt through DB lookup (1)
- * easy to start, hard in production
- * easier for real time and dynamic data
- * challenges:
  - * messy real world data
  - * inaccurate retrieval, complex questions
  - * risks need to be mitigated
EG: InternNad Dynamic knowledge base
* CAG = possibly better than RAG (8)
- * Requires High Context Window Model
- * Dont Need to worry about some RAG difficulties
- * Simpler
- * Gemini 2.0 Flash models have improved things a lot to allow this, hallucination rate is near 0

* Finetuning = complicated process to do properly (2)
- * static
- * specialized tasks
- * mimic "personality"
- * cheaper & faster
- - * [Process]
- - 1. Prepare dataset process
>> IMPROVE DATASET
- - 2. together.ai or fireworks ai (to get started). RunPod or Modal to train & deploy own
+ + 3. Consider cost & speed for inference. 3B-8B is good for small and fast
- - 4. Consider the purpose. Specialized task? => Specialized model (HuggingFace)
+ + 5. Full-finetune = Low-rank adaptation LoRA
+ + 6. Use unsloth to train faster
+ + 7. Update training data to model specific syntax
>> PRINT AND UNDERSTAND A LITTLE MORE
- - 8. BAD RESULT DEBUGGING:
- - - * big model usually needs bigger datasets, may need more data
- - - * more reasoning ability? maybe need bigger model
- - 9. Export model once satisfied with results
>> AFTER OKAY MODEL WE EXPORT
- - - TERMS: 
- - - - quantization: fitting higher bits into lower bits, smaller model but less precise tradeoff
- - - - r: how many params impact? 16 good start (then 32,64...)
- - - - target module: which piece do you want to finetune. putting all is good start
- - - - lora_alpha: higher = more impact the lora will finetune will have, balance of under/over-fitting
EG: InternNad Personality
EG: Maybe specific domain knowledge?

* Web scraping with AI (3,4)
- * structured data extraction
EG: Pipeline to feed RAG

* Using Cursor AI workflow (4,5) or Claude 3.7 (11)
- * (5) Quickly create nice UI example
- * Start with Claude as usecase may be good enough (Fuds cursor in this vid pretty much fudded 4/5)!
- * State of the art prob and a lot faster

* ComfyUI kek (6) or Gemini 2.0 (10)
- * Ecom product creations
- * Start with Gemini 2.0 as usecase may be good enough!

* MCP (9)
- * External systems plugged into AI



* Deep research o3-mini (i.e. searches and gives result back) (7)
- * Real time detailed data