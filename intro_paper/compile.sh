#!/bin/bash

pweave -i noweb -f tex mparse_intro.pnw
pdflatex mparse_intro.tex
bibtex mparse_intro.aux
pdflatex mparse_intro.tex
pdflatex mparse_intro.tex
texcount mparse_intro.tex