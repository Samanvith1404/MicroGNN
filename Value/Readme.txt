(Note: THIS FILE HAS BEEN TAKEN WITH THE REFERENCE OF ANDREJ AUTOGRADING (MICROGRAD))

What this file is about

Value.py is a tiny implementation of automatic differentiation.
It basically acts like the “math engine” behind everything else in the project.

Whenever we do an operation like addition, multiplication, tanh, or exponentials, this class keeps track of where each value came from. Later, when we call .backward(), it walks back through all those operations and computes gradients.

It’s the same idea that libraries like PyTorch use internally, but obviously much smaller and easier to follow.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Why I added this file

Instead of depending on a big library just to get gradients, I wanted something simple and transparent that I fully understand.
This class lets me build neural-network-style calculations from scratch, which is helpful for debugging and experimenting.

Every weight in the GNN uses this class, so training the model is possible only because this file keeps track of gradients.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

What it actually does

1)Stores a number (data)
2)Stores its gradient (grad)
3)Remembers which previous values created it
4)Knows which operation produced it
5)Has a .backward() method to push gradients through the whole graph


It supports basic operations needed for neural networks:

1)+
2)-
3)*
4)powers
5)tanh
6)exp

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

