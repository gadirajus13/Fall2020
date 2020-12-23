ALL IMAGES ARE LOCATED IN PDF DOCUMENT
Dataset was taken from 2006 Middlebury Stereo Datasets

Pictures 1:

I ran region-based analysis with Sum of Absolute Differences as the matching score. I then ran this for 3 levels with 
a template size of 10 and matching score of 100. This resulted in images that roughly made out the shape of the baby but did not pick up any facial
expressions. However, it did accurately recognize the shape of the baby without considering the box. The right disparity was also a lot more 
accurate than the left disparity as it better recognizes the left hand side of the body.

Pictures 2:

I ran a feature-based analysis with Sum of Absolute Differences as the matching score. I then ran this for 3 levels with 
a template size of 7 and matching score of 100. This resulted in images that were similar to Pictures 1, however the right disparity of this one
did a much better job of recognizing the legs and the feet than the region-based analysis.

Pictures 3:

I ran a region-based analysis with Sum of Squared Differences as the matching score. Similar to the ones before, I ran this at 3 levels with
a template size of 7 and a window size of 100. This gave me images very similar to region-based with Sum of Absolute differences except
these images seemed a little worse. The left disparity did not clearly pick up any feauture except the face and part of the left hand.
However, the right disparity was a lot more accurate, getting the entirty of the body inlcuding the legs and some of the feet.

Pictures 4:

I ran a feature based analysis with Sum of Squared Differences as the matching score. I ran this at 3 levels with a template size of 7 and
window size of 100. Although these are the darkest images produced by far, they do not do a good job of clearly showing any of the features, as
you cannot make out any major body parts or the shape of the baby without actively looking for it.

Pictures 5:

I ran region-based analysis with normalized cross-correlation as the matching score. Then I ran it at 4 levels with a template size of 6 and a matching
window size of 10. Thsi resulted in images that seeme random and do not detect any features between either of the disparities. I am guessing this is due to a problem in my calculation for the disparity while using NCC.
