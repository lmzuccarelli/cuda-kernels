Day 98: Progress Checkpoint – Easy Theoretical Quiz

Below are 10 easy-level LeetCode-style theoretical questions along with detailed solutions. These questions recap advanced debugging, host multithreading, CUDA graphs, and precision topics covered in Days 85–97. Each question is accompanied by a comprehensive Mermaid diagram that illustrates the key concepts.
1. What is the purpose of advanced debugging in CUDA?

Answer:
Advanced debugging in CUDA is used to diagnose complex issues such as race conditions and deadlocks in multi-stream or multi-block applications. Tools like cuda-gdb, cuda-memcheck, and Nsight Systems help trace execution, inspect memory accesses, and verify that synchronization primitives are correctly applied.


2. Why is host multithreading important for GPU coordination?

Answer:
Host multithreading allows multiple CPU threads to concurrently manage CUDA streams and launch kernels. This increases throughput by overlapping CPU preparation and GPU execution, ensuring that the GPU remains fully utilized even when managing multiple tasks.


3. What are CUDA graphs and how do they benefit application performance?

Answer:
CUDA graphs allow you to capture a sequence of kernel launches and memory operations as a graph, which can be instantiated and replayed with minimal overhead. They reduce kernel launch latency and enforce explicit dependencies, thereby improving performance and predictability in repetitive workflows.


4. How does numerical precision affect iterative computations in GPU applications?

Answer:
Numerical precision determines the number of significant digits used in calculations. In iterative computations, small rounding errors can accumulate over many iterations, leading to significant inaccuracies. Choosing between float (lower precision) and double (higher precision) is crucial, and algorithms like Kahan summation can mitigate error accumulation.


5. What is the Kahan Summation Algorithm used for?

Answer:
The Kahan Summation Algorithm is used to reduce the error accumulation in the summation of floating-point numbers. It maintains a compensation variable to correct for small rounding errors at each addition, leading to a more accurate final result


6. What role does occupancy play in kernel performance?

Answer:
Occupancy refers to the ratio of active warps on an SM to the maximum number of warps supported. High occupancy helps hide memory latency by allowing the GPU to switch between warps. However, maximizing occupancy without considering other factors (such as instruction-level parallelism) may not always lead to optimal performance.


7. What is the significance of using pinned memory in data transfers?

Answer:
Pinned (page-locked) memory enables faster data transfers between the host and device by allowing direct memory access (DMA). It also supports asynchronous transfers, which can overlap with kernel execution, thereby reducing overall data transfer latency.


8. What is a cooperative group and how does it aid synchronization?

Answer:
Cooperative groups are an API that allows threads to form groups beyond the traditional block level, including grid-level synchronization. They enable fine-grained control of synchronization, which is essential for complex collective operations across multiple blocks.


9. How does GPU-to-GPU (P2P) communication benefit multi-GPU applications?

Answer:
GPU-to-GPU communication using P2P allows direct data transfers between GPUs without involving the host, reducing latency and CPU overhead. This is particularly useful in multi-GPU and multi-node systems for scaling applications and improving data throughput.


10. Why is it important to use CUDA graphs for repetitive workflows?

Answer:
CUDA graphs capture a series of operations into a single, replayable entity, drastically reducing kernel launch overhead for repetitive tasks. They provide deterministic execution and allow efficient replays of complex workflows, which is crucial for iterative applications and real-time processing.
