## Using QLoRA on the new Phi-3 from microsoft and testing it.

edit1: I'll merge without quantization later. Just testing stuff and memory caps. 

edit2: merged adapter both with and without quantization (careful as I'm not reseting gpu memory throughout the model declarations, so some of it might go to RAM and slow things down)
___
___

#### Quick context on LoRA, regardless of the framework one is working in:

**Rank (R)**: Measure of "dimensionality" of the space a transformation - weight matrix - can span. 

The max rank is always the min(#rows, #columns)... why? because that's the max rank possible when assuming all rows and columns are linearly independent.

Higher rank means more parameters to learn, more memory, more energy spent on concept/semantic space.
___
**Low (Lo)**: One wants to approximate a subset of these weight matrices (x layers) with lower rank matrices (check SVD or other factorization methods). [A]

After that, typically a simple linear projection gets us back to the correct dimension that must match the next input dimension. [B]

The rest of the network has its weights frozen during the forward and backward pass. 
___
**Adaptation (A)**: One adapts the pre-trained model to a more specific task/objective, capturing important relationships from this new environment, comparatively to the innitial general model training.
___
**LoRA**: One fine-tunes a large network not simply by freezing some weights and re-training on specific data, but actually reducing the computational cost by performing and working with low rank approximation matrices. $inDim \times outDim$ -> $rank \times (inDim + outDim)$

![image](https://github.com/seyeint/Fine_tuning_torch/assets/36778187/a0430c2e-aa0b-4754-909e-3d8ad37b2349)

For an intuitive overview of rank and alpha parameters, check [this](https://medium.com/@fartypantsham/what-rank-r-and-alpha-to-use-in-lora-in-llm-1b4f025fd133). 
