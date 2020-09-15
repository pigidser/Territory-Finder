# Territory prediction for a trade outlet

This project uses https://github.com/cdrx/docker-pyinstaller.git repository that provides a container for ease compiling Python applications to binaries files (exe).

### How to compile the project:
1. Put modified python scripts in /src folder (main.py, territory_finder.py, etc.)
2. Go to the /src folder
$ cd src
3. Build executables
Linux
$ docker run -v "$(pwd):/src/" cdrx/pyinstaller-windows
Windows (be sure /src folder is shared - Docker Settings > Resources > File Sharing)
> docker run -v "%cd%:/src/" cdrx/pyinstaller-windows 

It will build your PyInstaller project into dist/windows/
