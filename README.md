# UDA-DP
The project for paper: UDA-DP: Unsupervised Domain Adaptation for Software Defect prediction



Software defect prediction can automatically locate defective code modules to focus testing resources better. Traditional defect prediction methods mainly focus on manually designing features, which are input into machine learning classifiers to identify defective code. However, there are mainly two problems in prior works. First manually designing features is time consuming and unable to capture the semantic information of programs, which is an important capability for accurate defect prediction. Second the labeled data is limited along with severe class imbalance, affecting the performance of defect prediction. In response to the above problems, we first propose a new unsupervised domain adaptation method using pseudo labels for defect prediction(UDA-DP). Compared to manually designed features, it can automatically extract defective features from source programs to save time and contain more semantic information of programs. Moreover, unsupervised domain adaptation using pseudo labels is a kind of transfer learning, which is effective in leveraging rich information of limited data, alleviating theproblem of insufficient data. Experiments with 10 open source projects from the PROMISE data set show that our proposed UDA-DP method outperforms the state-of-the-art methods for both within-project and cross project defect predictions. Our code and data are available at https://github.com/xsarvin/UDA-DP.



## I. Requirements



To install all library:

```
$ pip install -r requirements.txt
```



## II. Usage:

```
$ python run_uda.py
```

