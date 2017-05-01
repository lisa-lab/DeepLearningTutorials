This repository contains a framework to load the most commonly used datasets
for image and video semantic segmentation. The framework can perform some
on-the-fly preprocessing/data augmentation, as well as run on multiple threads
(if enabled) to speed up the I/O operations.

If you use this code, please cite:
* \[1\] Francesco Visin, Adriana Romero - Dataset loaders: a python library to
    load and preprocess datasets ([BibTeX](
        https://gist.github.com/fvisin/7104500ae8b33c3b65798d5d2707ce6c#file-dataset_loaders-bib))

### How to install it:
1. Clone the repository with `--recursive` in some path, e.g. to your `$HOME`:

   ```sh
   git clone --recursive https://github.com/fvisin/dataset_loaders.git "$HOME/dataset_loaders"
   ```

2. Add that path to your `$PYTHONPATH` (replace `$HOME/dataset_loaders` with
   the path you cloned into):

   ```sh
   echo 'export PYTHONPATH=$PYTHONPATH:$HOME/dataset_loaders' >> ~/.bashrc
   ```

3. The framework assumes that the datasets are stored in some *shared paths*,
   accessible by everyone, and should be copied locally on the machines that
   run the experiments. The framework automatically takes care for you to copy
   the datasets from the *shared paths* to a *local path*. 

   Create a configuration file with these paths in 
   `/dataset_loaders/dataset_loaders/config.ini` (see the 
   [config.ini.example](dataset_loaders/config.ini.example) in the same 
   directory for guidance).

   Note: if you want to disable the copy mechanism, just specify the same path 
   for the local and the shared path:

   ```ini
   [general]
   datasets_local_path = /a/local/path
   [camvid]
   shared_path = /a/local/path/camvid
   [cityscapes]
   shared_path = /a/local/path/cityscapes/
   (and so on...)
   ```


4. To use the MS COCO dataset, you also need to do the following:

   ```sh
   cd dataset_loaders/images/coco/PythonAPI
   make all
   ```
4. You will need to install SimpleITK if you intend to use the *warp_spline*
   data augmentation:

   ```sh
    pip install SimpleITK --user  
   ```
</br>

### Notes
* **The code is provided as is, please expect minimal-to-none support on it.**
* This framework is provided for research purposes only. Although we tried our 
  best to test it, the code might be bugged or unstable. Use it at your own
  risk!
* The framework currently supports image or video based datasets. It could be 
  easily extended to support other kinds of data (e.g., text corpora), but
  there is no plan on our side to work on that at the moment.
* Feel free to contribute to the code with a PR if you find bugs, want to
  improve the existing code or add support for other datasets.

 
</br>
</br>
</br>

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
