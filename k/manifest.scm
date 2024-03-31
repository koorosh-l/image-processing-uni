(use-modules (gnu packages image-processing)
	     (gnu packages python)
	     (gnu packages python-science)
	     (gnu packages machine-learning))
(packages->manifest `(,opencv
		      ,python
		      ,python-scikit-image
		      ,python-scikit-learn
		      ,python-scikit-learn-extra
		      ,tensorflow))
