# Overview

https://www.tensorflow.org/lite/examples/bert_qa/overview 

Needs to support following:

```
Inputs:
        input_ids: Int32[1, 384]
        input_mask: Int32[1, 384]
        segment_ids: Int32[1, 384]
Outputs:
        end_logits: Float32[1, 384]
        start_logits: Float32[1, 384]
```