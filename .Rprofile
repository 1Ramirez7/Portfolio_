source("renv/activate.R")
# Force reticulate to use local .venv Python instead of global PATH or WindowsApps version
# I'm not excluding out of git push since no sensitive info on here yet
venv_path <- file.path(getwd(), ".venv", "Scripts", "python.exe")
if (file.exists(venv_path)) {
  Sys.setenv(RETICULATE_PYTHON = venv_path)
}

Sys.setenv(QUARTO_PYTHON = "C:/git/Portfolio_/.venv/Scripts/python.exe")

tinytex_bin <- "C:/git/Portfolio_/.venv/TinyTeX/bin/win64"
Sys.setenv(PATH = paste(tinytex_bin, Sys.getenv("PATH"), sep = .Platform$path.sep))

