# Impresso Multilingual Text Embedder

This repository offers tools for embedding texts in multiple languages with an efficient workflow. It uses the `transformers` library by Hugging Face and the `make` tool to manage large datasets. `Make` ensures robust and incremental processing, allowing you to handle new data, resume tasks, and run processes across different machines, all while avoiding redundant work.

## Features

- **Efficient Storage Management:** Minimal local storage is required as necessary data for each year of a newspaper are downloaded on-the-fly and truncated after uploading.
- **Parallel Processing:** Processes run in parallel to optimize throughput.
- **Selective Processing:** Only the necessary processing steps are executed, ensuring efficiency by not reprocessing existing outputs on S3.
- **S3 Integration:** Integration with S3 for storing and resuming processing. The system ensures no overwriting of files or partial uploads due to interruptions.
- **Custom Embedding Options:** Flexible configurations via normal environment variables or make variables, including the ability to specify model versions and filter text data.

### Missing Features

- Batch processing of texts is not yet implemented. This will be added in a future
  release.
- Installing specialized xformer implementation for sparse attention inference is not yet
  imlemented. This shoulld be added in a future release for faster inference.

### Concepts

### Storage Locations

- **Local Storage:** Temporary disk space used for processing tasks. This disk space can
  be fast storage.
- **S3 Storage:** Permanent storage where final results are stored. Processing can be resumed from this storage.

### File Stamps

To manage dependencies, local file stamps are used:

- **Input Stamps (`.stamp`):** Indicate the status of input files.
- **Output Stamps (`.done`):** Indicate the completion of output files.

Local stamps help `make` determine which files need to be processed or skipped. The helper script `lib/sync_s3_filestamps.py` manages these stamps by syncing them with S3.

### File Organization

The processing follows a structured organization:

- **S3 Buckets:** Data is organized by processing steps and, in some cases, versions.
- **Build Directory:** A local mirror of the S3 storage, structured similarly for consistency.

```plaintext
# Example directory structure
BUILD_DIR/BUCKET/NEWSPAPER/<NEWSPAPER-YEAR>.jsonl.bz2
BUILD_DIR/BUCKET/PROCESSING_TYPE/VERSION/NEWSPAPER/<NEWSPAPER-YEAR>.jsonl.bz2
```

## Setup

1. **Clone the Repository:**

   ```bash
   git clone git@github.com:impresso/impresso-text-embedder.git
   cd impresso-text-embedder
   ```

Copy the `dotenv.sample` file to `.env` and modify the .env file to set the `AWS_ACCESS_KEY_ID` and `AWS_SECRET`.

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

---

<p align="center">
  <img src="https://github.com/impresso/impresso.github.io/blob/master/assets/images/3x1--Yellow-Impresso-Black-on-White--transparent.png?raw=true" width="350" alt="Impresso Project Logo"/>
</p>
