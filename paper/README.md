# Manuscript source

`main.tex` is the cleaned final source. The bibliography is already expanded
in `references_inline.tex`, so BibTeX is not needed. Compile from the repository
root with `make paper`, or run `latexmk -pdf main.tex` in this directory.

The five tables included by `main.tex` live in `tables/`. Three document fixed
inputs and priors; the PCA/evidence and Wilks tables are generated from the
analysis CSVs with `make tables`.

`manuscript.pdf` is the final PDF supplied with the source. To create a compact
journal-upload archive containing only referenced files, run `make submission`.
