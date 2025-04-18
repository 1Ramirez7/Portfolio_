


Commit # 70

I started using the R studio renv package
- This project has its on library folder and r dependencies similar to the python .venv virtual folder. 
- visit this site for instrucitons: https://rstudio.github.io/renv/
Also like a commit or two ago I made a python .venv folder and added .Rprofile and .Renviron so the project uses that python folder for dependencies and not others outside the project root folder. 
- added the tinytex dependencies to the .venv folder and included in the project env .Rprofile and .Renviron

Commit # 71

Deleted automatic files and re render site

commit # 79

I ran into list rendering issues so I remove projects files not structure for the new listing format. It seems to have fix the issue. I also switch my portfolio_ name to 1ramirez7.github.io because that is what I'm supposed to use for my portfolio

commit # 85 

I did rename my portfolio and I think that is what cause the issue because my github url change so I needed to also update it on R Studio

commit # 87 

I'm still having that issue.

I tested removing the about section, remove added freeze. I don't know what can be causing this issue.


commit # 89 

commit 88 fix it, I ran a whole bunch of clean cache things. did not chamge any code that wasnt already change before. 
I noticed on some of the json and other file changes, when I added new listing folders, since I name them the same main.qmd I noticed github was not picking up the whole file change.so it say i only change projects/china to projects/rdus but kept the old projects/china/main.qmd.  I probably need tp have different names for them to avoid id issues. 


commit # 90

I'm going to move away from having same name files, and I will make a _metadata.yml to specify listing names for each project. 
This allows to keep unique names and allows to have other qmd files in that folder. 