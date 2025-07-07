# Project Journal - Health Monitoring System

Title: Bluebox
Author: Jerry Huang and Nathan Yan
Description: Fully functional diagnostic device which detects all the patient's vitals in less than 10 seconds!
Created_at: 2025-07-07

## Total Time Spent: 16 hours

## July 3rd - Thursday
**Time Spent:** 4 hours

**What We worked on:**
This was our first time trying out the raspberry pi so We were learning to connect the entire thing. We reached out to some people who helped me configure the raspberry pi via ethernet to someone's router. We were able to ssh into our raspberry pi which was very cool! Towards the evening, We were able to connect the Pi Camera Module 3 to our pi and ran some detection code on it. We offloaded the detection to a GPU which analyzed the image and outputted text using LLaVA, an open source image-to-text model.

**Pictures/Videos:**
[Progress Photos + Videos/July3rd.JPEG](Progress%20Photos%20+%20Videos/July3rd.JPEG)
[Progress Photos + Videos/July3rd.MOV](Progress%20Photos%20+%20Videos/July3rd.MOV)

## July 4th - Friday
**Time Spent:** 4 hours

**What We worked on:**
I spent some time connecting our MLX90614 temperature sensor to our raspberry pi. I was able to run some basic code on it which allowed it to detect ambient and object temperature. I also integrated an AI doctor into our image-to-text model which my friend helped create. Essentially, the text from the image-to-text model would be inputted as additional patient information into an AI diagnoser model which would eventually lead to a diagnosis. Towards the end of the day, I also tried designing a hand placement device which would use the

**Pictures/Videos:**
[Progress Photos + Videos/July4th.JPEG](Progress%20Photos%20+%20Videos/July4th.JPEG)

## July 5th - Saturday
**Time Spent:** 4 hours

**What We worked on:**
I finished the hand placement device prototype on Fusion. We also added supporting beams and a case to put the raspberry pi beneath. Towards the end of the day, I received the MAX30105 and was fiddling around with some code to properly amplify the signals received from the MAX30105 to detect pulse, SpO2 and Respiratory Rate. 

**Pictures/Videos:**
[Progress Photos + Videos/July5th.mov](Progress%20Photos%20+%20Videos/July5th.mov)

## July 6th - Sunday
**Time Spent** 4 hours

**What We worked on:**
We worked on the software for all the components. Because We have not bought the components yet, We are not completely certain the the software will function as accordingly. However, We are confident because We have done our research in it. 

**Pictures/Videos:**
No pictures and videos since today was all software. 