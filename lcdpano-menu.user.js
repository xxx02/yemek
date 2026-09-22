// ==UserScript==
// @name         LCDPano Sabit Yemek Menüsü
// @namespace    https://yemekliste.netlify.app/
// @version      1.1.0
// @description  Yemek menüsü aktifken onu LCDPano animasyonundan ayırıp sabit gösterir; başka modüle geçildiğinde otomatik gizler.
// @match        https://app.lcdpano.net/*
// @run-at       document-idle
// @grant        none
// @updateURL    https://raw.githubusercontent.com/xxx02/yemek/main/lcdpano-menu.user.js
// @downloadURL  https://raw.githubusercontent.com/xxx02/yemek/main/lcdpano-menu.user.js
// ==/UserScript==

(function () {
  'use strict';

  const MENU_URL = 'https://yemekliste.netlify.app/menu.svg';
  const OVERLAY_ID = 'lcdpano-fixed-menu-overlay';
  const ANCHOR_ATTR = 'data-lcdpano-menu-anchor';

  let anchor = null;
  let overlay = null;
  let overlayImage = null;
  let rectRatio = null;
  let stableTimer = null;
  let currentDayKey = '';

  const style = document.createElement('style');
  style.textContent = `
    #${OVERLAY_ID} {
      position: fixed !important;
      z-index: 2147483000 !important;
      display: none;
      margin: 0 !important;
      padding: 0 !important;
      border: 0 !important;
      overflow: hidden !important;
      background: #fdfdfd !important;
      pointer-events: none !important;
      transform: translateZ(0) !important;
      backface-visibility: hidden !important;
      -webkit-backface-visibility: hidden !important;
      contain: layout paint style !important;
    }

    #${OVERLAY_ID} img {
      display: block !important;
      width: 100% !important;
      height: 100% !important;
      max-width: none !important;
      max-height: none !important;
      margin: 0 !important;
      padding: 0 !important;
      border: 0 !important;
      object-fit: contain !important;
      object-position: left top !important;
      background: #fdfdfd !important;
    }

    img[${ANCHOR_ATTR}="1"] {
      opacity: 0 !important;
    }
  `;
  document.head.appendChild(style);

  function onSettingsPage() {
    return location.pathname.startsWith('/settings');
  }

  function getDayKey() {
    const parts = new Intl.DateTimeFormat('tr-TR', {
      timeZone: 'Europe/Istanbul',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit'
    }).formatToParts(new Date());

    const out = {};
    for (const part of parts) {
      if (part.type !== 'literal') out[part.type] = part.value;
    }

    return `${out.year}-${out.month}-${out.day}`;
  }

  function getMenuSrc() {
    return `${MENU_URL}?day=${getDayKey()}`;
  }

  function findAnchor() {
    const candidates = [...document.images].filter(img => {
      const src = img.currentSrc || img.src || '';
      return src.includes('yemekliste.netlify.app/menu.svg');
    });

    return candidates.find(img => isAnchorActive(img)) || null;
  }

  function validRect(rect) {
    return (
      Number.isFinite(rect.left) &&
      Number.isFinite(rect.top) &&
      rect.width >= 80 &&
      rect.height >= 50 &&
      rect.right > 0 &&
      rect.bottom > 0 &&
      rect.left < window.innerWidth &&
      rect.top < window.innerHeight
    );
  }

  function isAnchorActive(img) {
    if (!img || !img.isConnected) return false;

    const rect = img.getBoundingClientRect();
    if (!validRect(rect)) return false;

    let el = img.parentElement;
    while (el && el !== document.documentElement) {
      const cs = getComputedStyle(el);
      if (
        cs.display === 'none' ||
        cs.visibility === 'hidden' ||
        parseFloat(cs.opacity || '1') < 0.05
      ) {
        return false;
      }
      el = el.parentElement;
    }

    const x = Math.min(window.innerWidth - 1, Math.max(0, rect.left + rect.width / 2));
    const y = Math.min(window.innerHeight - 1, Math.max(0, rect.top + rect.height / 2));
    const top = document.elementFromPoint(x, y);

    if (!top) return false;

    const face = img.closest('.face, .front, .back, [class*="slide"], [class*="carousel"]');
    if (face) {
      return top === face || face.contains(top);
    }

    return top === img || top.contains(img) || img.contains(top);
  }

  function nearlyEqual(a, b, tolerance = 1.5) {
    return (
      Math.abs(a.left - b.left) <= tolerance &&
      Math.abs(a.top - b.top) <= tolerance &&
      Math.abs(a.width - b.width) <= tolerance &&
      Math.abs(a.height - b.height) <= tolerance
    );
  }

  function saveRect(rect) {
    rectRatio = {
      left: rect.left / window.innerWidth,
      top: rect.top / window.innerHeight,
      width: rect.width / window.innerWidth,
      height: rect.height / window.innerHeight
    };
  }

  function applyRect() {
    if (!overlay || !rectRatio) return;

    overlay.style.left = `${rectRatio.left * 100}vw`;
    overlay.style.top = `${rectRatio.top * 100}vh`;
    overlay.style.width = `${rectRatio.width * 100}vw`;
    overlay.style.height = `${rectRatio.height * 100}vh`;
  }

  function ensureOverlay() {
    if (overlay && overlay.isConnected) return;

    overlay = document.createElement('div');
    overlay.id = OVERLAY_ID;

    overlayImage = document.createElement('img');
    overlayImage.alt = 'Günlük yemek menüsü';
    overlayImage.draggable = false;

    overlay.appendChild(overlayImage);
    document.body.appendChild(overlay);
  }

  function refreshMenu(force = false) {
    ensureOverlay();

    const dayKey = getDayKey();
    if (!force && dayKey === currentDayKey && overlayImage.src) return;

    const nextSrc = getMenuSrc();
    const preload = new Image();

    preload.onload = () => {
      overlayImage.src = nextSrc;
      currentDayKey = dayKey;
      applyRect();

      if (anchor && isAnchorActive(anchor)) {
        anchor.setAttribute(ANCHOR_ATTR, '1');
        overlay.style.display = 'block';
      } else {
        overlay.style.display = 'none';
      }
    };

    preload.onerror = () => {
      console.warn('[LCDPano Menü] SVG yüklenemedi:', nextSrc);
    };

    preload.src = nextSrc;
  }

  function captureStableAnchor(img) {
    if (stableTimer) clearInterval(stableTimer);

    let lastRect = null;
    let stableCount = 0;
    let attempts = 0;

    stableTimer = setInterval(() => {
      attempts += 1;

      if (!img.isConnected) {
        clearInterval(stableTimer);
        stableTimer = null;
        return;
      }

      const rect = img.getBoundingClientRect();

      if (!validRect(rect)) {
        if (attempts >= 40) {
          clearInterval(stableTimer);
          stableTimer = null;
        }
        return;
      }

      if (lastRect && nearlyEqual(rect, lastRect)) {
        stableCount += 1;
      } else {
        stableCount = 0;
      }

      lastRect = {
        left: rect.left,
        top: rect.top,
        width: rect.width,
        height: rect.height
      };

      // Yaklaşık 1 saniye aynı konumda kaldığında koordinatı kilitle.
      if (stableCount >= 4) {
        saveRect(lastRect);
        applyRect();
        refreshMenu(true);

        clearInterval(stableTimer);
        stableTimer = null;

        console.info('[LCDPano Menü] Sabit konum kilitlendi.', lastRect);
      }
    }, 200);
  }

  function attach() {
    if (onSettingsPage()) {
      if (overlay) overlay.style.display = 'none';
      return;
    }

    const found = findAnchor();

    if (!found) {
      if (overlay) overlay.style.display = 'none';

      if (anchor && anchor.isConnected) {
        anchor.removeAttribute(ANCHOR_ATTR);
      }

      anchor = null;
      return;
    }

    if (anchor !== found) {
      if (anchor && anchor.isConnected) {
        anchor.removeAttribute(ANCHOR_ATTR);
      }

      anchor = found;
      captureStableAnchor(anchor);
    }

    // LCDPano aynı resmi DOM'da yeniden oluşturursa orijinali tekrar gizle.
    if (rectRatio && anchor.isConnected && isAnchorActive(anchor)) {
      anchor.setAttribute(ANCHOR_ATTR, '1');
      applyRect();
      refreshMenu(false);

      if (overlayImage && overlayImage.src) {
        overlay.style.display = 'block';
      }
    } else if (overlay) {
      overlay.style.display = 'none';
    }
  }

  const observer = new MutationObserver(attach);
  observer.observe(document.documentElement, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ['src', 'class', 'style']
  });

  window.addEventListener('resize', () => {
    applyRect();
  });

  window.addEventListener('pageshow', () => {
    attach();
    refreshMenu(false);
  });

  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
      attach();
      refreshMenu(false);
    }
  });

  // LCDPano'nun slayt/menü rotasyonunu takip et.
  // Yemek modülü ekrandan çıktığında overlay gizlenir; geri geldiğinde yeniden görünür.
  setInterval(() => {
    attach();
  }, 150);

  // Gün değişimini ayrıca düşük maliyetle kontrol et.
  setInterval(() => {
    refreshMenu(false);
  }, 30 * 1000);

  attach();
})();
