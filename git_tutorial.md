# 有用处的Git命令

```bash
git config --global user.name "Your Name"
git config --global user.email "email@example.com"
```

## Repository

什么是版本库？版本库又名仓库，英文名repository,你可以简单的理解一个目录，这个目录里面的所有文件都可以被Git管理起来，每个文件的修改，删除，Git都能跟踪，以便任何时刻都可以追踪历史，或者在将来某个时刻还可以将文件”还原”

```{bash}
cd C:
cd Users
cd 16000 #user name
cd Documents
cd DNN_project
pwd #check the path
```

通过命令 git init 把这个目录变成git可以管理的仓库，如下：

```{bash}
git init
Initialized empty Git repository in C:/Users/16000/Documents/DNN_project/.git/
```

或者，右键->git bash here

### Add & commit

```bash
git add file1.txt
git add file2.txt file3.txt
git commit -m "add 3 files."
#add 3 files is a submit anotation
```



### 查看

```{bash}
git status
```

如果对文件做了修改但是并未提交，那么就会有提示，

通过 `git diff`命令查看修改，

```{bash}
git diff readme.txt
```

```{bash}
git log #查看日志
git reflog #查看命令历史
git reset --hard commit_id #回到之前版本
```



### ls/cat/mv/touch/rm

ls

```{bash}
ls -1 #每行列出一个文件，即以单列形式列出
ls -a #所有文件
```
cat
```{bash}
cat readme.txt #打印文字
cat file1 file2> target_file #合并文件
```
mv
```{bash}
mv file1.txt file2.txt #改名字

```

touch

```{bash}

```

rm

```bash
git rm test.txt
git commit -m "remove test.txt"
```

## 工作区



## Github

```{bash}
ssh-keygen -t rsa -C "youremail@example.com"
ssh-add id_rsa
```

### push

```bash
git remote add origin git@github.com:com3dian/DNN_project.git #com3dian is my account
git push -u origin master #type yes!!!!!

#之后只要
git push origin master
```

### clone

```{bash}
git clone git@github.com:com3dian/DNN_project.git

```







