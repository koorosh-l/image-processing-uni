#! /bin/sh
# -*- mode: scheme; coding: utf-8 -*-
exec guile -e main -s "$0" "$@"
!#
(use-modules (aiscm core)
	     (aiscm image)
	     (aiscm magick)
	     (aiscm filters)
	     (ice-9 match))
(define (to-gray img) (from-image (convert-image (to-image img) 'GRAY)))

(define (roberts-cross img)
  (define (norm x y) (/ (+ (abs x) (abs y)) 2))
  (/ (norm (convolve img (arr (+1 0) (0 -1)))
	   (convolve img (arr (0 +1) (-1 0)))) 2))

(define (sobel img)
  (define (norm x y) (/ (+ (abs x) (abs y)) 8))
  (norm (convolve img (arr (1 0 -1) (2 0 -2) (1 0 -1)))
	(convolve img (arr (1 2 1)  (0 0 0)  (-1 -2 -1)))))

(define (gg-filter img)
  (define (norm x y) (sqrt (+ (* x x) (* y y))))
  (norm (gauss-gradient-x img 2.0)
	(gauss-gradient-y img 2.0)))

(define (edges img method)
  (let ([img (to-gray (read-image img))])
    (match method
      ['roberts-cross (roberts-cross img)]
      ['sobel         (sobel img)]
      ['gg-filter     (gg-filter img)]
      [else (error "unknown method" method)])))
(define (main args)
  (match args
    [(name method in out)
     (write-image (to-type <ubyte> (edges in (string->symbol method)))
		  out)]
    [else (error "args should be input output" args)]))
