## Project Overview
The goal in monocular depth estimation is to predict the depth value of each pixel or to infer general depth information, given only one single RGB image as input. In the example picture above, the leftmost column is RGB images, and the middle row is depth information maps of the corresponding images.

As you may have experienced, this task isn’t so difficult for human eyes, but it still remains somewhat challenging for robots’ perception modules. It’s not hard to imagine the value depth estimation can bring: autonomous vehicles can be more aware of their distance to obstacles, factory robots can know better about the things they manipulate, etc. Roboticists in industry and academia care about this question a lot.

## Project Logistics
Team Members
This is a group project. You can have a team of up to 3 persons. On the other hand, this project is also designed to be solo-able, so if you feel like it, you can totally try to do it all by yourself. But please be aware that we will not lower our requirements just because you are soloing it!

You can find your team member in the #self-introduction-spring23 channel of our Slack space.
Dataset
We will use NYU-Depth V2 dataset. Please don’t download the dataset directly from this website for now. 

Instead, please use this link to download an enhanced dataset that includes more datapoints
Due Date
You can submit the project any time after Week 5. The submission will close on 05/09/2023. Before the due date, you can submit unlimited times.
Deliverables
You should submit a zip folder containing all the necessary code to run your project and a 2-3 pages report documenting your thought process and how to properly run your project.

More technical details incoming

## Evaluation
Your project will be evaluated on a test partition of NYU-Depth. We will also base a small portion of your grade on the report.

It will also be evaluated on a hidden dataset to check for generalizability. If we find a huge disparity between performance on the NYU-Depth test set and the hidden dataset, and at the same time, if your code contains parts where you intentionally try to hardcode the results/let the model memorize all NYU-Depth data, then we will have to give you a lower score, even though it is not technically cheating.

There are a few metrics that we will evaluate your methods on.
Assuming that dgt and dp are the ground truth and predicted depths at a given pixel of any testing image, and T denotes a collection of pixels that the ground truth values are available. 

Note that the metrics below will be calculated over the entire test set. So realistically, T will be all the pixels of all the pictures in the test set
Absolute Relative Difference (abs rel)

[![image.png](https://i.postimg.cc/yxkL3qw3/image.png)](https://postimg.cc/bSfx45Yq)

Here |T| indicate cardinality of T (so the number of elements in T), whereas |dgt-dp| means the absolute value of their difference

[![image.png](https://i.postimg.cc/DZDMgyQP/image.png)](https://postimg.cc/JDNx782s)

Here max() is a function that takes the maximum value of the two inputs. Threshold is usually set to be 1.25

Comparison between Metrics
RMSE captures a scenario where being off by 10 is more than twice as bad as being off by 5, whereas in abs rel, being off by 10 is exactly twice as bad as being off by 5.

In abs rel, the error is relative to the magnitude of the ground truth itself. Basically, if the magnitude of the ground truth is large, then off by a small amount is no big deal.

We also use threshold because we prefer a scenario where all the predicted depth info is relatively accurate over a scenario where some predictions are extremely accurate, but some are extremely off.

## Leaderboard
This project will feature an in-class leaderboard based on your project performance. Every time you submit the project, you can choose to participate in the leaderboard, either using your real name, a team name, or anonymously. 

Placing in the top 3 of the leaderboard will earn you huge extra credits, and if you would like to, you may consider developing your project into a serious publication or combine it with some other ongoing research projects in the AI4CE lab. [I am considering a small 3d-printed souvenir for the top 1?]

Of course, placing at the very bottom of the leaderboard will not induce any penalty. We will not grade your project solely based on the leaderboard. This is mainly for fun. Don’t take this too seriously.

How to Approach This Problem
### Step 0
We will introduce more technical details of this problem when we talk about epipolar geometry during Week 5 of this class. But before that, there are still things that can be done to help you be more prepared for this project.

Basics and Intuition of Depth Estimation
This Medium article is a great starting point to get some basic intuition of this problem. You don’t need to get every detail of it, but reading it will definitely help you. If you can’t access it, let me know, and I have a .pdf version of it.

Write a Dataloader
When working with real computer vision/artificial intelligence/deep learning problems, we usually don’t get the luxury that the dataset is pre-built in PyTorch.

Instead, in most cases, we will have to download the dataset ourselves, usually in a folder of png files, and will need to write code to organize them into a nice PyTorch Dataloader object. 

However, this process is not really that intuitive and needs some practice to get used to. This PyTorch tutorial covers the details of how to proceed with this. I suggest you try out this tutorial before proceeding with the project

### Step 1: Designing A Loss Function

GPU Computing Resources

If you decide to pursue a project that is heavy on deep learning, then you will need a GPU (Graphic Processing Unit). 

If you have a discrete Nvidia GPU in your computer, then please refer to this Slack thread on how to configure your local environment. Unfortunately, the TA will not be able to provide any support for AMD GPUs (and AMD’s ROCm platform)

If you do not have a GPU, then we can offer the following support for GPU computing resources. (Listed from most recommended to the least)

Free Methods
Google Colab
As used in the hands-on lectures. Notebooks from hands-on lectures 1 and 2 cover the basics of how to use Colab.
Pros: 
The free version can provide up to 12 consecutive hours of free GPU computation per session.
Almost no configuration is needed
Entirely free

Cons
The computation available is relatively limited compared to other methods listed in the document.
The GPU time can be terminated abruptly before the 12-hour limit depending on the traffic at Google’s data centers.
Colab has no persistent storage by default. Every time you fire up a Colab session, you start with empty storage. So every time, you will need to re-upload the dataset and remember to download any results as soon as they are available.
This can be partially resolved by mounting a Google Drive folder to your Colab session. This is introduced in hands-on lecture 1’s notebook.


Google Cloud VM-Backed Colab
IMPORTANT: This method is free in principle but can cost you money if operated incorrectly. The university or any member of the teaching staff will not be responsible for any unwanted cost incurred during your use of this service.

Google Cloud actually gives free credit for every new account, and it’s enough to support you through this entire project. 
Pros:
Very powerful computing
Basically free, with some caveats
Cons:
Complicated to set up
May accidentally cost you money if you don’t operate it correctly
Will ask you for payment info, even though Google won’t charge you

If you are interested in this method, please follow this tutorial. Please read it carefully and thoroughly. it may cost you money if you are not operating it correctly.


NYU HPC
NYU provides a very powerful high-performance computing cluster for students to use. If you intend to use this method, please contact TA directly. You will be given further instructions on how to proceed.

Pros:
Powerful computation with almost no major limitations in terms of computation time and storage
Entirely free

Cons:
The learning curve on how to configure and use it can be steep. 
Your priority in the HPC system may be low, so your program may experience a long wait time before it can be executed.


Paid Service
There are also relatively cheap GPU renting services out there that you may utilize. But we recommend trying out the listed free methods first before deciding to pay for such a service. Again, paying for a service doesn’t automatically translate to getting a higher score in this class, and we do not recommend you do this just for the sake of completing this class project. 

Google Colab Pro
You can upgrade Colab either through a monthly subscription or a pay-as-you-go scheme. You can access Google’s documentation here. Google also has a blog explaining the pay-as-you-go scheme here.

IMPORTANT: Please read Google’s documentation carefully before you proceed to purchase this service. The university or any member of the teaching staff will not be responsible for any unwanted cost incurred during your use of this service.

Lambda Lab
Lambda Lab is a company that specializes in providing GPU access for deep learning tasks. You can find their website here.

IMPORTANT: Please read Lambda Lab’s documentation carefully before you proceed to purchase this service. The university or any member of the teaching staff will not be responsible for any unwanted cost incurred during your use of this service.


## FAQ
Q: Why is there so much missing information incoming?
A: This project is a brand new iteration for this class. While the bulk of it should be working, I am still in the process of developing and testing it so I can make your project experience smoother. Thank you so much for your patience.
On the other hand, there is some required background knowledge that won’t be covered until Week 5, so I don’t expect many students will be able to work much before that anyway, so students are not losing much time (hopefully)

Q: Instead of “Monocular,” what are some other depth estimation schemes?
A: Besides monocular, there are binocular or stereo depth estimations. We will talk about them in more detail in Week 5, but as you may have already guessed, estimating depth info with only one image is much more challenging than doing that with multiple images

Q: What’s stopping me from using ChatGPT to finish this project?
A: ChatGPT is not an all-knowing oracle. It’s optimized as a conversation AI and is best at keeping the conversation going. To this end, it may provide very useful information, or it may make things up completely. So I wouldn’t blindly trust it. 
On another note, ChatGPT is not capable of autonomously finishing tasks this complex without any serious human supervision/intervention, for now. So you may end up spending more time prompting ChatGPT than actually doing it yourself

Q: What’s stopping me from just grabbing an open-source project or an open-source paper on Github?
A: The professors and TA of this class basically make a living by monitoring progress on these types of robotics questions. If you just grab a Github project and don’t put in any effort, it will be very easy for us to spot.
Besides, we were all students once as well, so we know the game of “changing a few lines of code to make it look like I am not copying homework” as well.
So don’t try this. We are not gonna make the grading of this project super harsh anyway, so try to enjoy it, and it may actually turn out to be a fun project. 
