exports.handler = async function () {
  const TIME_ZONE = 'Europe/Istanbul';

  const menus = {
    '2026-09-21': ['Et Döner / Patates', 'Pirinç Pilavı', 'Ayran'],
    '2026-09-22': ['Etli Çorba', 'Taze Fasulye', 'Mantı'],
    '2026-09-23': ['Püreli Misket Köfte', 'Bulgur Pilavı', 'Komposto'],
    '2026-09-24': ['Kuru Fasulye', 'Tavuklu Pirinç Pilavı', 'Ayran'],
    '2026-09-25': ['Piliç Külbastı', 'Mac and Cheese', 'Mevsim Meyve'],
    '2026-09-28': ['Piliç Baget Fırın', 'Tel Şehriyeli Pirinç Pilavı', 'Ayran'],
    '2026-09-29': ['Kaşarlı Domates Çorba', 'Çiftlik Kebabı', 'Bulgur Pilavı'],
    '2026-09-30': ['Ev Usulü Mercimek Çorba', 'Biber Dolma', 'Gül Börek'],
    '2026-10-01': ['Minestrone Çorba', 'Pizza / Patates Kızartması', 'Ayran'],
    '2026-10-02': ['İzmir Köfte', 'Peynirli Makarna', 'Mevsim Meyve'],
    '2026-10-05': ['Nohut Yemeği', 'Pilav', 'Cacık'],
    '2026-10-06': ['BBQ Soslu Sebzeli Et Sote', 'Naneli Bulgur Pilavı', 'Vişne Komposto'],
    '2026-10-07': ['Terbiyeli Köfte', 'Spagetti Napoliten', 'Ballı Badem Tatlısı'],
    '2026-10-08': ['Tavuk Dürüm / Patates', 'Pirinç Pilavı', 'Ayran'],
    '2026-10-09': ['Etli Düğün Çorba', 'Kış Türlüsü', 'Fırın Makarna'],
    '2026-10-12': ['Chicken Stroganoff', 'Sebzeli Kuskus Makarna', 'Sütlaç'],
    '2026-10-13': ['Kaşarlı Köfte', 'Nohutlu Pilav', 'Elma Komposto'],
    '2026-10-14': ['Şehriyeli Tavuk Suyu Çorba', 'Kıymalı Ispanak', 'Cevizli Erişte'],
    '2026-10-15': ['Mercimek Çorbası', 'Sebzeli Meksika Usulü Dana Eti (Acısız)', 'Bulgur Pilavı'],
    '2026-10-16': ['Kuru Fasulye', 'Pirinç Pilavı', 'Ayran'],
    '2026-10-19': ['Teriyaki Soslu Piliç', 'Üç Renkli Sebzeli Bulgur Pilavı', 'Mevsim Meyve'],
    '2026-10-20': ['Kremalı Brokoli Çorba', 'Yufkalı Bohça Kebabı', 'Arpa Şehriye Pilavı'],
    '2026-10-21': ['Nohut', 'Tavuklu Pilav', 'Ayran'],
    '2026-10-22': ['Dalyan Köfte', 'Peynirli Kuskus', 'Sonbahar Salata'],
    '2026-10-23': ['Karnabahar Graten', 'Soslu Makarna', 'Browni'],
    '2026-10-26': ['Lahmacun', 'Ayran', 'Salata'],
    '2026-10-27': ['Yıldız Şehriyeli Domates Çorbası', 'Yayla Soslu Piliç', 'Kekikli Bulgur Pilavı'],
    '2026-10-28': ['Patates Graten', 'Burgu Makarna', 'Akdeniz Salata'],
    '2026-10-30': ['Mevsim Türlü', 'Mantı', 'Yoğurt'],
    '2026-11-02': ["Süzme Mercimek Çorbası", "Soslu Tavuk Döner Lavaş", "Kaşık Patates", "Ayran"],
    '2026-11-03': ["Közlenmiş Domates Çorbası", "Püreli Hasanpaşa Köfte", "Tel Şehriyeli Bulgur Pilavı", "Mevsim Meyve"],
    '2026-11-04': ["Etli Düğün Çorba", "Kuru Fasulye", "Tavuklu Pilav", "Cacık"],
    '2026-11-05': ["Tarhana Çorba", "Dana Bulgogi", "Penne Makarna", "Mevsim Yeşillikleri"],
    '2026-11-06': ["Ezogelin Çorba", "Karnabahar Graten", "Arpa Şehriye Pilavı", "Supangle"],
    '2026-11-09': ["Naneli Yoğurt Çorba", "Gemici Usulü Nohut", "Pirinç Pilav", "Cacık"],
    '2026-11-10': ["Arpa Şehriye Çorba", "Dana Fajita", "Erişte", "Red Velvet"],
    '2026-11-11': ["Domates Çorba", "Soya Soslu Piliç", "Lazanya", "Yoğurt"],
    '2026-11-12': ["Saray Çorba", "Sebze Graten", "Naneli Bulgur Pilavı", "Vişne Komposto"],
    '2026-11-13': ["Tavuk Suyu Çorba", "Kuru Köfte / Patates", "Şeh. Pilav", "Kış Salatası"],
    '2026-11-23': ["Toyga Çorba", "Piliç Schnitzel", "Renkli Biberli Bulgur Pilavı", "Elma Komposto"],
    '2026-11-24': ["Şeh. Tavuk Çorba", "Kaşarlı Köfte", "Spagetti", "Havuç Tarator"],
    '2026-11-25': ["Etli Düğün Çorba", "Kuru Fasulye", "Şeh. Pirinç Pilavı", "Cacık"],
    '2026-11-26': ["Ev Usulü Mercimek Çorba", "Hamburger", "Patates Kızartması / Soğan Halkası", "Browni"],
    '2026-11-27': ["Tarhana Çorbası", "Mengen Musakka", "Mısırlı Pilav", "Mevsim Meyve"],
    '2026-11-30': ["Ezogelin Çorba", "Etli Patates Yemeği", "Pey. Boncuk Makarna", "Mevsim Salata"],
  };

  function getDateInfo(now = new Date()) {
    const parts = new Intl.DateTimeFormat('tr-TR', {
      timeZone: TIME_ZONE,
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
    }).formatToParts(now);

    const map = {};
    for (const p of parts) {
      if (p.type !== 'literal') map[p.type] = p.value;
    }

    const title = new Intl.DateTimeFormat('tr-TR', {
      timeZone: TIME_ZONE,
      weekday: 'long',
      day: 'numeric',
      month: 'long',
    }).format(now);

    return {
      key: `${map.year}-${map.month}-${map.day}`,
      title,
    };
  }

  function escapeXml(value) {
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&apos;');
  }

  function wrapText(text, maxChars = 28) {
    const words = String(text).split(/\s+/);
    const lines = [];
    let current = '';

    for (const word of words) {
      const trial = current ? `${current} ${word}` : word;
      if (trial.length <= maxChars) {
        current = trial;
      } else {
        if (current) lines.push(current);
        current = word;
      }
    }

    if (current) lines.push(current);
    return lines.length ? lines : [''];
  }

  const today = getDateInfo();
  const menu = menus[today.key];
  const isNovemberBreak = today.key >= '2026-11-16' && today.key <= '2026-11-20';
  const width = 360;
  const xBullet = 16;
  const xText = 30;
  const contentWidth = 320;

  let y = 58;
  let content = '';

  if (menu && menu.length) {
    for (const item of menu) {
      const lines = wrapText(item, 28);
      const bulletY = y - 10;
      content += `<rect x="${xBullet}" y="${bulletY}" width="7" height="7" rx="1" transform="rotate(45 ${xBullet + 3.5} ${bulletY + 3.5})" fill="#d35400" />`;
      lines.forEach((line, idx) => {
        content += `<text x="${xText}" y="${y + idx * 18}" font-size="18" font-family="system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif" fill="#333" font-weight="500">${escapeXml(line)}</text>`;
      });
      y += lines.length * 18 + 10;
    }
  } else if (isNovemberBreak) {
    content += `<text x="16" y="60" font-size="18" font-family="system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif" fill="#d35400" font-weight="600">16–20 Kasım: Ara Tatil</text>`;
    content += `<text x="16" y="86" font-size="16" font-family="system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif" fill="#7f8c8d">Yemek servisi bulunmamaktadır.</text>`;
    y = 108;
  } else {
    content += `<text x="16" y="60" font-size="18" font-family="system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif" fill="#7f8c8d" font-style="italic">Bugün yemek servisi bulunmamaktadır.</text>`;
    y = 86;
  }

  const height = Math.max(110, y + 10);
  const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title desc">
  <title id="title">${escapeXml(today.title)} Menüsü</title>
  <desc id="desc">Ali Öztaylan Anadolu İmam Hatip Lisesi günlük yemek menüsü</desc>
  <rect width="100%" height="100%" fill="#fdfdfd"/>
  <text x="16" y="24" font-size="20" font-family="system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif" fill="#d35400" font-weight="700">${escapeXml(today.title)} Menüsü</text>
  <line x1="16" y1="34" x2="${contentWidth + 16}" y2="34" stroke="#eee" stroke-width="2"/>
  ${content}
</svg>`;

  return {
    statusCode: 200,
    headers: {
      'Content-Type': 'image/svg+xml; charset=utf-8',
      'Cache-Control': 'no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0',
      'Pragma': 'no-cache',
      'Expires': '0',
      'Surrogate-Control': 'no-store',
      'X-Content-Type-Options': 'nosniff',
      'Access-Control-Allow-Origin': '*',
    },
    body: svg,
  };
};
