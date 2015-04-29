.PHONY: all clean clean-all install dist-all dist install-dist

VERSION=0.2.18
DISTUSER=sojka
DISTDIR=aisa:/www/lemma/projekty/download
DISTNAME=fithesis2-v$(VERSION)
DISTFILES=$(DISTNAME).pdf $(DISTNAME).zip
CLASSFILES=fit1[012].clo fithesis.cls fithesis[23].cls
AUXFILES=example.aux example.log example.out example.toc example.lot example.lof fithesis.aux fithesis.log fithesis.toc fithesis.ind fithesis.idx fithesis.out fithesis.ilg fithesis.gls fithesis.glo fi-logo600.514pk fi-logo600.600pk fi-logo600.tfm
PDFFILES=fithesis.pdf example.pdf
LOGOFILES=loga/phil-logo.eps loga/med-logo.pdf loga/ped-logo.pdf loga/med-logo.eps loga/sci-logo.eps loga/fsps-logo.pdf loga/fss-logo.pdf loga/fsps-logo.eps loga/law-logo.eps loga/ped-logo.eps loga/sci-logo.pdf loga/law-logo.pdf loga/fss-logo.eps loga/econ-logo.eps loga/econ-logo.pdf loga/phil-logo.pdf
SOURCEFILE=fithesis.dtx
FILOGOFILES=fi-logo.mf fi-logo600.mf
OTHERFILES=csquot.sty $(FILOGOFILES) example.tex fithesis.ins Makefile tutorial.pdf
INSTALLFILES=$(CLASSFILES) $(LOGOFILES) $(PDFFILES) $(SOURCEFILE) $(OTHERFILES)
TEXLIVEFILES=$(CLASSFILES) $(LOGOFILES)

# Tento pseudocíl vytvoří soubory třídy, příklad,
# dokumentaci a následně odstraní pomocné soubory.
all: fithesis2.cls $(PDFFILES) clean

# Tento cíl vytvoří soubory třídy.
fithesis2.cls: fithesis.ins fithesis.dtx
	tex $<

# Tento cíl vysází dokumentaci.
fithesis.pdf: fithesis.dtx
	pdflatex $<
	makeindex -s gind.ist fithesis
	makeindex -s gglo.ist -o fithesis.gls fithesis.glo
	pdflatex $<
#	pdflatex $<

# Tento cíl vysází příklad.
example.pdf: example.tex fithesis2.cls
	pdflatex $<
	pdflatex $<

# Tento pseudocíl instaluje veškeré nepomocné soubory
# do adresáře určeného parametrem "to".
install:
	@if [ -z "$(to)" ]; then echo "Usage: make to=DIRECTORY install"; exit 1; fi
	mkdir --parents "$(to)/fithesis2"
	cp --parents --verbose $(INSTALLFILES) "$(to)/fithesis2"

# Tento pseudocíl instaluje soubory tříd a technickou
# dokumentaci do adresářové struktury balíku TeXLive,
# jejíž kořenový adresář je určen parametrem "to".
install-texlive:
	@if [ -z "$(to)" ]; then echo "Usage: make to=DIRECTORY install-texlive"; exit 1; fi
	mkdir --parents "$(to)/texmf-local/tex/latex/fithesis2"
	cp --parents --verbose $(TEXLIVEFILES) "$(to)/texmf-local/tex/latex/fithesis2"
	mkdir --parents "$(to)/texmf-local/doc/latex/fithesis2"
	cp fithesis.pdf "$(to)/texmf-local/doc/latex/fithesis2/manual.pdf"
	mkdir --parents "$(to)/texmf-local/fonts/source/filogo"
	cp $(FILOGOFILES) "$(to)/texmf-local/fonts/source/filogo"
	texhash

# Tento pseudocíl odstaní pomocné soubory.
clean:
	rm -f $(AUXFILES)

# Tento pseudocíl odstraní veškeré vytvořitelné soubory.
clean-all: clean
	rm -f $(PDFFILES) $(CLASSFILES)

# Tento pseudocíl připraví distribuci, zveřejní ji
# a smaže lokální soubory distribuce.
dist-all: dist install-dist dist-clean

# Tento pseudocíl připraví distribuci.
dist: $(DISTFILES) 

$(DISTNAME).zip: $(INSTALLFILES)
	zip -r -v $(DISTNAME).zip $(INSTALLFILES)

$(DISTNAME).pdf: fithesis.pdf
	cp fithesis.pdf $(DISTNAME).pdf

# Tento pseudocíl uveřejní distribuci verze $(VERSION)
install-dist: $(DISTFILES)
	scp $(DISTFILES) $(DISTUSER)@$(DISTDIR)

# Tento pseudocíl odstraní lokální soubory distribuce
dist-clean:
	rm -f $(DISTFILES)

