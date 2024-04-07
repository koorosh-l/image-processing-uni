(use-modules (gnu packages image-processing) ;;packages
	     (gnu packages guile-xyz)
	     (gnu packages guile)
	     (gnu packages python)
	     (gnu packages image)
	     (gnu packages python-science)
	     (gnu packages pkg-config)
	     (gnu packages machine-learning)
	     (gnu packages texinfo)
	     (gnu packages texlive)
	     ;;guixes
	     (guix licenses)
	     (guix utils)
	     (guix packages)
	     (guix download)
	     (guix git-download)
	     (guix build-system gnu))

(define-public guile-cv
  (package
   (name "guile-cv")
   (version "0.4.0")
   (source (origin
	    (method url-fetch)
	    (uri "http://ftp.gnu.org/gnu/guile-cv/guile-cv-0.4.0.tar.gz")
	    (sha256 (base64 "60dx0jRoSe/SIsvI//XIxpvF/hf3B7uP5rTnUPsHwgA="))))
   (build-system gnu-build-system)
   (arguments
    `(#:phases
      (modify-phases %standard-phases
		     (add-after 'unpack 'prepare-build
				(lambda* (#:key inputs outputs #:allow-other-keys)
				  (substitute* "configure"
					       (("SITEDIR=\"\\$datadir/guile-cv\"")
						"SITEDIR=\"$datadir/guile/site/$GUILE_EFFECTIVE_VERSION\"")
					       (("SITECCACHEDIR=\"\\$libdir/guile-cv/")
						"SITECCACHEDIR=\"$libdir/"))
				  (substitute* "cv/init.scm"
					       (("\\(dynamic-link \"libvigra_c\"\\)")
						(string-append "(dynamic-link \""
							       (assoc-ref inputs "vigra-c")
							       "/lib/libvigra_c\")"))
					       (("\\(dynamic-link \"libguile-cv\"\\)")
						(format #f "~s"
							`(dynamic-link
							  (format #f "~alibguile-cv"
								  (if (getenv "GUILE_CV_UNINSTALLED")
								      ""
								      ,(format #f "~a/lib/"
									       (assoc-ref outputs "out"))))))))
				  (setenv "GUILE_CV_UNINSTALLED" "1")
				  ;; Only needed to satisfy the configure script.
				  (setenv "LD_LIBRARY_PATH"
					  (string-append (assoc-ref inputs "vigra-c") "/lib"))
				  #t)))))
   (inputs
    (list vigra vigra-c guile-3.0 texinfo))
   (native-inputs
    `(("texlive" ,texlive)
      ("pkg-config" ,pkg-config)))
   (propagated-inputs
    `(("guile-lib" ,guile-lib)))
   (home-page "https://www.gnu.org/software/guile-cv/")
   (synopsis "Computer vision library for Guile")
   (description "Guile-CV is a Computer Vision functional programming library
for the Guile Scheme language.  It is based on Vigra (Vision with Generic
Algorithms), a C++ image processing and analysis library.  Guile-CV contains
bindings to Vigra C (a C wrapper to most of the Vigra functionality) and is
enriched with pure Guile Scheme algorithms, all accessible through a nice,
clean and easy to use high level API.")
   (license gpl3+)))


(packages->manifest `(,coreutils
		      ,opencv ,tensorflow
		      ,python
		      ,python-scikit-image ,python-scikit-learn ,python-scikit-learn-extra
		      ,guile-3.0
		      ,guile-aiscm ,guile-cv))
