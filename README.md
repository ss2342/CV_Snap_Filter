# CV Final Project: Snapchat 
## Abstract: 

The creation and use of images that fit over a detected person’s face, or filter images, have been becoming increasingly popular in this age of social media. These filters that were popularized by Snapchat and currently used in other social media apps such as Messenger, and Instagram leverage Computer Vision and Image Processing concepts in order to properly align a base filter image over a face. This report will explore these concepts, and using these techniques in Python, a basic but original implementation of “Snapchat” filters was created and will be discussed in this report.



## Instructions to Run our SnapChat Filter Code:
1. Download a copy of the notebook from Either the Github Repo or the Colab Notebook listed below:
   - **Google Colab:**
    https://colab.research.google.com/drive/12pKhzuU_ifBvpWIeais21oSAr0UgmOB2?usp=sharing
    
2. Make a copy of the Google Colab Notebook or import the .ipynb file into Google Colab
3. Run the first line of the newly created google colab notebook, this should download all the necessary files 
4. Run through all the cells (unless you do not want to train the model) before the Testing out SCFilterOverlay() Function header to set up all the functions and models 
   1. **IF you want to avoid training the model**, skip the cell that is under the header **Start Training** and set the variable near the top of the code called `pretrained = true` , otherwise set this to false. We have uploaded our own file for the weights called “Best_weights.hdf5”  to circumvent waiting for training to take place. This file is downloaded using the `!gdown` function.


5. Run the code under **Testing out SCFilterOverlay() Function**. The following parameters can be changed within the cell block with the `### CHANGE VARIABLES IN THIS CELL ### ` comment to change the image, filter, or model path:
   1. Test_img_path: change this variable to point to the path of the image you want to feed into the algorithm
     1. You can upload your own images if you drag your desired image into the contents folder into Google Colab( on the left). We recommend you use images that are larger than size 96x96 pixels that has a close up view of a face
     2. We provide some test images to try out in the folder Test_Images within the github. To reference these photos, copy this path into test_img_path: ‘./CV_Snap_Filter/Test_Images/**'INSERTIMAGENAMEHERE’**


   2. model_path: specifies file that contains parameters of trained model. The default is `‘./Best_weights.hdf5’` due to the training model code saving the file in that directory, but if you are uploading another version of this file, you have to specify the path that the parameter file is currently in


   3. filter: To make life simpler, we just have the user insert a number 0-7 to call a filter. Listed below will be a table of the integers to filter names. 
      1.** Table of Filter Images **
	- Filter PNG Name Description
	- 0 Luigi_Stache Mustache with Nose 
	- 1 HP_Glasses Round glasses
	- 2 Eyes Googly Eyes
	- 3 Binky A Binky
	- 4 Mario_Hat A hat
	- 5 beard A beard
 	- 6 blush Anime-esque blush that goes on cheeks
	- 7 squidward_nose An elongated nose similar to Squidward’s from the show Spongebob Squarepants
	- 8 clown_nose Clown nose

     2. If you want to apply MULTIPLE filters, there are a few steps that have to be taken
       1. Step 1: create a new filter variable under the original filter variable
       2. Call SCFilterOverlay(), but the image pushed into this function should be the image that has the previous filter overlaid already
         1. Example : 
     
            `new_im = SCFilterOverlay(test_img,lm,filter)
	    
            new_im2 = SCFilterOverlay(new_im,lm,filter1)`
     3. You can add your own filter images if you follow the following steps
       1. Upload an image into the Filter_Images Folder on google collab (or github but you have to reclone the github in google collab)
       2. Mark down desired landmarks on filter image and state what these landmarks correspond to in the createLandMarks() function
       3. Add a new if statement with a new index number and follow the format of the previous if statements in SCFilterOverlay()
