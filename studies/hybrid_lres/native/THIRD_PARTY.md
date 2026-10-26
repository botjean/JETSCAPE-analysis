# Third-Party Source Notices

Only LundGenerator.cc/.hh and their supplied GPL license text are included in
`vendor/`. No heppy source files, checkout, binary, or environment are distributed.

* EXTERNAL ONLY: GroomerShop.cc/.hh and fjtools.cxx/.hh from matplo/heppy commit
  `16fcefbdca7a106e6e74fd53e904e302335dcce3`. Their license is not established
  here; this package does not distribute them or assign them a license. Their
  source paths and exact SHA256 are dependency records in SOURCE_MANIFEST.json.
  Optional heppy-dependent builds read a separately supplied pinned checkout.
* LundGenerator.cc/.hh: fjcontrib 1.051 LundPlane, with the executed heppy header
  patch. Preserve the file copyright and GPL version 2-or-later notices.
  `COPYING.LundPlane` is the supplied GPL license text. This patched header, not
  an arbitrary system Lund header, is part of the executed source identity.

The source manifest gives exact origin paths and hashes. This notice is not a
claim that all code in a linked extension can be redistributed under BSD.
