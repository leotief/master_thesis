# the paper: main.tex -> main.pdf (pdflatex + bibtex, as the icml template)
MAIN = main

pdf:
	pdflatex -interaction=nonstopmode $(MAIN)
	-bibtex $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc

.PHONY: pdf clean
