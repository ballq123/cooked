Belle Connaught (bconnaug)
11-685 Introduction to Deep Learning
HW1P2 Code & Ablations Log Submission

WANDB PROJECT LINK: 
    https://wandb.ai/bconnaug-carnegie-mellon-university/hw1p2?nw=nwuserbconnaug 


HOW TO RUN MY CODE:
    I wrote all of my code in a single notebook, which is named "hw1p2_s25_starter_notebook.ipynb" 
and is included in this handin.tar zip file. To run it, one would simply open the notebook in
Google Colab (and ideally connect to some custom GCE VM so the model doesn't take forever to train), 
then run each code block sequentially by either manually pressing the play button on each code block one 
at a time, or by clicking Runtime > Run all.  


DISCLAIMERS:
    1 - In my last iteration of training, I was trying to 'squeeze' my model to see if my
validation or train accuracy could increase any more before the final deadline. Therefore, 
in the "Experiment" section, I manually set config['epochs'] = 1 and ran my model for one
epoch at a time, so that I would not accidentally be training past the deadlinee. So, 
running my code as-is would cause it my model to train for only one epoch.

    2 - Similarly, towards the end of my training, I found my model was working pretty well. 
Thus, rather than starting brand-new WanDB runs each time, I initialized one and simply
resumed it after each training iteration, updating my learning rate to be the latest epoch's
learning rate (which changed dynamically since I used the ReduceLROnPlateau scheduler) in the 
"Experiment" section. For this reason, my latest run (as shown by my WanDB graphs) is 210 epochs long. 

    3 - Around epochs 70 and 196, there are strange dips in the WanDB training/validation 
accuracy graphs. This is because when resuming the runs, I accidentally forgot to update the
learning rate properly, essentially 'setting my model back' a couple epochs.

    4 - The "config" dictionary is not entirely accurate, as I manually adjusted variables
such as the model archetype, learning rate, scheduler, and epochs in the "Optimizer" and "Experiments"
sections of the notebook as the model trained.

    5 - Lastly, for all training sessions, I used a custom GCE VM with a Nvidia T4 GPU, as suggested by
Recitation 0.5 on the class website.


EXPERIMENTS & ARCHITECTURE DETAILS:
    I first tried upgrading my model from the provided (60% accuracy) model by
increasing the batch size to 1024, using the AdamW optimizer, xavier_normal weight initialization,
GELU activations, and increasing the depth of the diamond-shaped NN from 1 -> 15. This implementation 
was a decent improvement, as I trained my model on 50% of the dataset for 50 epochs and reached the 
following accuracies: 
        Train-70.1499%, Val-75.1369%, Test-75.359%

    I realized that my model was plateauing way too early (within the first 20 epochs), so 
I figured the learning rate should be able to change to avoid this early plateauing. Thus, 
I then defined a ReduceLROnPlateau scheduler (with factor = 0.5 and patience = 3) to dynamically 
adjust the learning rate as the model trained. I also decreased the weight decay to 0.01 to
reduce anticipated overfitting, increased the dataset to 75%, and decreased the batch size to 512
to increase generalization. Still training at 50 epochs, this yielded the following accuracies,
a satisfying jump from the previous iterations:
        Train-77.31865%, Val-82.4766%, Test-82.587%

    I then tried to deepen my network even more to have 20 layers and increased the 
patience of the scheduler from 3 -> 5, as I thought the learning rate was changing a bit
too frequently. This run was not successful, however, as the first 13 epochs yielded
a "NaN" training loss (for some reason that I still cannot figure out). It also seemed
to plateau a bit earlier than the previous run, so I decided to change my approach 
by doing the following:

    1 - Using SiLU (Sigmoid Linear Unit) activation functions instead of GELU, because some
    research revealed that it works well in deeper neural networks, and is slightly more
    computationally efficient.
    2 - Changing to OneCycleLR scheduler with max_lr=0.01 and steps=50*len(train_loader),
    because I thought it would help to generalize better, and boost the accuracy so
    that it would converge more quickly (and to a higher accuracy value). 
    3 - Using kaiming_uniform weight initializer, which seems to be more compatible
    with the SiLU activation function

    The changes above yielded the following accuracies, which were sadly worse than the 
previous model:
        Train-77.27917%, Val-82.50799%, Test-82.542%

    The OneCycleLR scheduler was just to aggressive, so I first attempted to use it with 
ReduceLROnPlateau (by running it for the first 15 epochs, then allowing ReduceLROnPlateau to 
take over for the remaining epochs). I also decreased the total number of epochs to 35, just 
so I could see a sample of this model's perfomance. The WanDB outputs were similar to the model 
above, so I stopped training early and switched to just using ReduceLROnPlateau, with patience 
decreased from 5 -> 3. I also decreased the weight decay to 0.0001 and changed the shape of the 
neural network to be more shallow and hourglass-like, with 17 total layers.
    
    This decision was the best yet, increasing my accuracies to the following:
        Train-80.652%, Val-83.5999%, Test-83.325%

    When all 35 epochs elapsed, I found that the model was not necessarily plateauing
yet, so I increased the dataset to 100% and resumed the WanDB run, using the most recent
learning rate as the starting point and running it for 35 epochs at a time. The model began to
plateau quite obviously (and was beginnning to overfit, since the difference between the validation
and train accuracies was nearly 0.5%) between epoch 160-170, so I manually increased the learning rate 
from 0 to 0.0001, and continued on. 

    This was not the best decision, as both the train and validation accuracies dipped by 1% 
(from 84% and 85% to 83% and 84%, respectively), so I set the learning rate back to what it was 
prior to plateauting (0.0000012) and continued on. This was a great decision, as the validation
acccuracy remained at ~85.9%, but the training accuracy dropped to 84.21% (indicating that the model 
was overfitting less). 
    I then continued allowing the model to train until it reached the following final (and BEST) 
accuracies around epoch 210, near the project deadline:
        Train-84.331%, Val-85.986%, Test-85.922%

        
