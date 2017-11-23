ASSIGNMENT = assignment
LATEX = pdflatex
penultimateword = $(wordlist $(words $1),$(words $1), x $1)

all: $(ASSIGNMENT)

$(ASSIGNMENT): 
	$(LATEX) $(ASSIGNMENT).tex
	@mv $(ASSIGNMENT).pdf ../12-752_Assignment_$(call penultimateword,$(subst /, ,$(CURDIR))).pdf
	
clean:
	-rm -f *.aux *.log *.pdf  *.out *~
