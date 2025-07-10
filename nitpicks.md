# Big
* Writing all your code in a main function is too boring and constraining. 
  We should allow users to define classes or functions as well. PyG is a great example.
  Although more discussion is required here.
* while a vector is perfectly fine for current use. When we use GPU we 
  will have to write some GPU-CPU vector class for convenience.
* good work on the output converter. But I think we need more examples to design something
  better. Find many GNN datasets and look at the output format.

# Medium
* You are copying the graph object in the main function. Not ideal, pass a malloc'ed 
  pointer? or use move semantics.


