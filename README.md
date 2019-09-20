# tsvm
The svm and tsvm algorithms for brain-computer interface. Our algorithms are tested in Matlab 2015a.

First, you should execute read_BCI_III_DSIVa.m and read_BCI_IV_DSIIa.m to obtain a total set for each subject.
Then,  you can execute tsvm_classifier.m or isttsvm_classifier.m to compare the perfomance of different classifiers.

The directories are the following:

(1) Classifier:

tsvm_classifier.m: the script of SVM, TSVM-light, RTSVM, LDS, CCCP, and ITSVM.
isttsvm_classifier.m: the script of IST-TSVM 

(2) EEGdata:
a) BCI competition III data set IVa: You can download this set at http://www.bbci.de/competition/iii/.
b) BCI competition IV data set IIa: You can download this set at http://www.bbci.de/competition/iv/.

Then you can unzip them if needed, and copy each data set into the corresponding subdirectory in the "OriginalData" directory (located itself in "EEGdata"). 

To randomly select some samples as training set and the remaining samples as test set, we put all samples into a total set. 
Before you execute read_BCI_III_DSIVa.m or read_BCI_IV_DSIIa.m to obtain a total set for each subject, you should install Biosig.
The version of Biosig we used is biosig4octmat-3.1.0.


(3) functions:
Various usefull toolbox. Noted that before running matlab, you should install C compiler, for example: Microsoft Visual C++ 2013 Professional (C).

(4) GeneratedData:
The results were saved in this folder.

(5) Utilities:
Small matlab source code file.
