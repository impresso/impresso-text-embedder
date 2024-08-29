# Impresso Multilingual Text Embedder

This is a repository for embedding impresso texts in multiple languages.
It uses a minimal set of dependencies and is designed to be easy to use.
It is based on the `transformers` library by Hugging Face.
For running the embedder on our large dataset, it uses the make tool to parallelize the
process and orchestrate the different steps to ensure processing steps are run when
necessary.

The Makefile build system is organized to:

- require only a small local disk storage for the processing
- download the necessary models and data on the fly
- run the processing steps in parallel
- ensure that the costly processing steps are run only when necessary
- respects already processed files on remote s3 storage
- upload the correctly results to s3 storage without overwriting existing files, or
  without having broken files on s3 due to interrupted processing steps.

### Concepts

There are two storage places for this processing:

- limited local storage: the local disk storage where the processing is done
- s3 storage: the large remote storage where the results are stored and where the processing
  can be resumed from.

Make can only compute the dependencies of a target based on the files that are present
on the local machine. This means that we need to have a way to know which input file are
newer than the output files, and which output files are already present on s3 storage.
For this, we create local file stamps (empty files with the correct time stamp set) that
give make the necessary information to compute the dependencies.
We create these file stamps for the input files, and for the output files that are
already on s3.
Stamp input files end with `.stamp` and stamp output files end with `.done`. The
makefile rules are written in such a way that the output files are only created when
there is no .done file present, and the .done file is only created when the output file
created locally is newer than its dependencies.
The helper script `lib/sync_s3_filestamps.py` is used to create the local stamp files.

### File organization and processing protocol

The processing is organized in the following way:

- On s3 storage, the data is organized in buckets. Each bucket corresponds to some
  processing step.
- Some buckets have the newspaper folders immediately inside the bucket: BUCKET/NEWSPAPER. Others have
  an internal structure with BUCKET/PROCESSING_TYPE/VERSION/NEWSPAPER.
- In impresso, each newspaper is in a separate folder.
- In each newspaper folder, the data files are organized by year and typically end with
  `<NEWSPAPER-YEAR>.jsonl.bz2`.
- We keep the processed files in a build folder referred to as BUILD_DIR. The build
  folder is organized in the same way as the s3 storage it mirrrors.

```
# local mirror of s3 storage to be worked on
# without intermediate processing type and version
BUILD_DIR/BUCKET/NEWSPAPER/<NEWSPAPER-YEAR>.jsonl.bz2

# local mirror of s3 storage to be worked on
# with intermediate processing type and version
BUILD_DIR/BUCKET/PROCESSING_TYPE/VERSION/NEWSPAPER/<NEWSPAPER-YEAR>.jsonl.bz2
```

## Setup

Clone this repository:

```bash
git clone multilingual-text-embedder
cd multilingual-text-embedder
```

Modify the .env file to set the `AWS_ACCESS_KEY_ID` and `AWS_SECRET`.

To install the repository, you need to have `make` and `python3` installed on your system.

## About

### Impresso

[Impresso - Media Monitoring of the Past](https://impresso-project.ch) is an
interdisciplinary research project that aims to develop and consolidate tools for
processing and exploring large collections of media archives across modalities, time,
languages and national borders. The first project (2017-2021) was funded by the Swiss
National Science Foundation under grant
No. [CRSII5_173719](http://p3.snf.ch/project-173719) and the second project (2023-2027)
by the SNSF under grant No. [CRSII5_213585](https://data.snf.ch/grants/grant/213585))
and the Luxembourg National Research Fund under grant No. 17498891.

### Copyrights

Copyright (C) 2018-2024 The Impresso team.     
Contributors to this program include: [Simon Clematide](https://github.com/simon-clematide)

### License

This program is provided as open source under
the [GNU Affero General Public License](https://github.com/impresso/impresso-pyindexation/blob/master/LICENSE)
v3 or later.

___

<p align="center">
  <img src="https://github.com/impresso/impresso.github.io/blob/master/assets/images/3x1--Yellow-Impresso-Black-on-White--transparent.png?raw=true" width="350" alt="Impresso Project Logo"/>
</p>


