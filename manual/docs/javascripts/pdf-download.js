/* Inserts a locale-aware "Download PDF" button into the Material header.
   English visitors get qmaxent-manual-en.pdf; visitors under /ko/ get
   qmaxent-manual-ko.pdf. The files are produced by the docs.yml CI
   workflow and committed alongside the rendered site. */
(function () {
  function injectPdfButton() {
    var headerInner = document.querySelector(".md-header__inner") ||
                      document.querySelector(".md-header");
    if (!headerInner) { return; }

    // Detect locale from the path. mkdocs-static-i18n serves the EN
    // build at the manual root and the KO build under /manual/ko/.
    var isKorean = /\/manual\/ko(\/|$)/.test(window.location.pathname);
    var pdfFile  = isKorean ? "qmaxent-manual-ko.pdf"
                            : "qmaxent-manual-en.pdf";
    var label    = isKorean ? "PDF 다운로드"
                            : "Download PDF";

    // Compute the URL relative to the manual root. The manual is at
    // /qmaxent/manual/ on GitHub Pages; PDFs live at the same level.
    var match = window.location.pathname.match(/^(.*\/manual\/)/);
    var base  = match ? match[1] : "/manual/";
    var pdfUrl = base + pdfFile;

    var link = document.createElement("a");
    link.href = pdfUrl;
    link.className = "md-header__button md-icon qmaxent-pdf-link";
    link.title = label;
    link.setAttribute("aria-label", label);
    link.setAttribute("download", pdfFile);
    link.innerHTML =
      '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">' +
        '<path fill="currentColor" d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 ' +
        '2 2h12a2 2 0 0 0 2-2V8l-6-6Zm-1 7V3.5L18.5 9H13Zm-1 4v6l-2.5-2.5' +
        'L9 18l4 4 4-4-1.5-1.5L13 19v-6h-1Z"/>' +
      '</svg>' +
      '<span class="qmaxent-pdf-link__label">' + label + '</span>';

    // Insert before the GitHub repo link if we can find it.
    var repoLink = headerInner.querySelector(".md-header__source") ||
                   headerInner.querySelector('[data-md-component="repository"]');
    if (repoLink) {
      headerInner.insertBefore(link, repoLink);
    } else {
      headerInner.appendChild(link);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", injectPdfButton);
  } else {
    injectPdfButton();
  }
})();
