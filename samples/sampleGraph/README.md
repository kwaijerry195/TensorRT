# Demo to show tensorrt graph mode failure
## Background
In recommendation, there are many small kernels to be launched. Without cuda graphs, the performance is bounded by enqueue.

TensorRT uses cuda stream capture apis to run in graph mode. However, when i/o buffers' addresses change, we have to recapture the graph, which is very time-consuming.

## How to run this demo?
1. Compile the sample by following build instructions in [TensorRT README](https://github.com/NVIDIA/TensorRT/).

2.  Run the sample to build and run the demo.
	```
    path/to/sample_graph  --datadir=path/to/this/source/directory
	```

    If the binary output directory is in `build/out/sample_graph`, you can run the command in project root directory, like

    ```
    build/out/sample_graph --datadir=samples/sampleGraph
    ```

3.  Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
	```
    &&&& RUNNING TensorRT.sample_graph [TensorRT v101401] [b48] # build/out/sample_graph --datadir=samples/sampleGraph
    [I] Building and running a GPU inference engine for model.onnx
    [I] [TRT] [MemUsageChange] Init CUDA: CPU +49, GPU +0, now: CPU 54, GPU 28424 (MiB)
    [I] [TRT] ----------------------------------------------------------------
    [I] [TRT] Input filename:   samples/sampleGraph/model.onnx
    [I] [TRT] ONNX IR version:  0.0.11
    [I] [TRT] Opset version:    13
    [I] [TRT] Producer name:    demo
    [I] [TRT] Producer version:
    [I] [TRT] Domain:
    [I] [TRT] Model version:    0
    [I] [TRT] Doc string:
    [I] [TRT] ----------------------------------------------------------------
    [I] [TRT] [MemUsageChange] Init builder kernel library: CPU +521, GPU +8, now: CPU 773, GPU 28432 (MiB)
    [I] [TRT] Local timing cache in use. Profiling results in this builder pass will not be stored.
    [I] [TRT] Compiler backend is used during engine build.
    [I] [TRT] Detected 1 inputs and 1 output network tensors.
    [I] [TRT] Total Host Persistent Memory: 80 bytes
    [I] [TRT] Total Device Persistent Memory: 0 bytes
    [I] [TRT] Max Scratch Memory: 0 bytes
    [I] [TRT] Total Activation Memory: 0 bytes
    [I] [TRT] Total Weights Memory: 16 bytes
    [I] [TRT] Compiler backend is used during engine execution.
    [I] [TRT] Engine generation completed in 4.78 seconds.
    [I] [TRT] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 0 MiB, GPU 1 MiB
    [I] [TRT] Loaded engine size: 0 MiB
    [I] [TRT] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 0 (MiB)
    [I] Input [0]: [[0, 1]]
    [I] Input [1]: [[1, 2]]
    [I] inferNonGraph:
    [I] Output[0]: [[0, 1]]
    [I] Output[1]: [[1, 2]]
    [I] inferGraphSlow:
    [I] Output[0]: [[0, 1]]
    [I] Output[1]: [[1, 2]]
    [I] inferGraphFast:
    [I] Output[0]: [[0, 1]]
    [I] Output[1]: [[0, 0]]
    [E] Verify failed: idx = 1, (i, j) = (0, 0), output vs expected = 0 vs 1
    [E] inferGraphFast failed!
    &&&& FAILED TensorRT.sample_graph [TensorRT v101401] [b48] # build/out/sample_graph --datadir=samples/sampleGraph
	```

	This output shows that the inferNonGraph and inferGraphSlow PASSED, but inferGraphFast FAILED
