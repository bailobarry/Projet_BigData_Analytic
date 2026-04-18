#!/usr/bin/env python
"""Script pour convertir le guide Markdown en HTML (imprimable en PDF)."""

import markdown
from pathlib import Path


def convert_md_to_html(md_path: str, html_path: str) -> None:
    """Convertit un fichier Markdown en HTML stylisé, prêt à imprimer en PDF."""
    
    # Lire le fichier Markdown
    md_content = Path(md_path).read_text(encoding="utf-8")
    
    # Convertir en HTML
    html_content = markdown.markdown(
        md_content,
        extensions=["fenced_code", "tables", "toc"]
    )
    
    # Style CSS pour un beau rendu et impression PDF
    css = """
        @media print {
            body { font-size: 10pt; }
            pre { page-break-inside: avoid; }
            h1, h2, h3 { page-break-after: avoid; }
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 14px;
            line-height: 1.7;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
            background: #fff;
        }
        
        h1 { 
            color: #1a5f7a; 
            font-size: 2em; 
            margin-top: 40px; 
            border-bottom: 3px solid #1a5f7a;
            padding-bottom: 10px;
        }
        
        h2 { 
            color: #2d8bba; 
            font-size: 1.5em; 
            margin-top: 35px; 
            border-bottom: 2px solid #2d8bba; 
            padding-bottom: 8px; 
        }
        
        h3 { 
            color: #4a4a4a; 
            font-size: 1.2em; 
            margin-top: 25px; 
        }
        
        code {
            background: #f4f4f4;
            padding: 3px 8px;
            border-radius: 4px;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 0.9em;
            color: #d63384;
        }
        
        pre {
            background: #282c34;
            border-radius: 8px;
            padding: 16px;
            overflow-x: auto;
            font-size: 0.85em;
            color: #abb2bf;
        }
        
        pre code {
            background: none;
            padding: 0;
            color: #abb2bf;
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-size: 0.9em;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        
        th { 
            background: #2d8bba; 
            color: white;
            font-weight: 600; 
        }
        
        tr:nth-child(even) { 
            background: #f8f9fa; 
        }
        
        tr:hover {
            background: #e9ecef;
        }
        
        blockquote {
            border-left: 4px solid #2d8bba;
            margin: 20px 0;
            padding: 15px 25px;
            background: #f8f9fa;
            font-style: italic;
        }
        
        ul, ol { 
            margin: 15px 0; 
            padding-left: 30px; 
        }
        
        li { 
            margin: 8px 0; 
        }
        
        strong { 
            color: #1a5f7a; 
        }
        
        hr {
            border: none;
            border-top: 2px solid #e9ecef;
            margin: 30px 0;
        }
        
        a {
            color: #2d8bba;
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        
        /* Table des matières */
        .toc {
            background: #f8f9fa;
            padding: 20px 30px;
            border-radius: 8px;
            margin: 20px 0;
        }
        
        /* Emoji support */
        .emoji {
            font-size: 1.2em;
        }
    """
    
    # HTML complet
    full_html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Lot A - Guide Complet pour Réécrire le Code</title>
    <style>
{css}
    </style>
</head>
<body>
<div class="print-instructions" style="background: #fff3cd; padding: 15px; border-radius: 8px; margin-bottom: 30px; border: 1px solid #ffc107;">
    <strong>📄 Pour créer le PDF :</strong> Appuyez sur <code>Ctrl+P</code> (ou <code>Cmd+P</code> sur Mac) et sélectionnez "Enregistrer au format PDF".
</div>

{html_content}

<footer style="margin-top: 50px; padding-top: 20px; border-top: 2px solid #e9ecef; color: #6c757d; font-size: 0.9em;">
    <p>📚 <strong>ELOQUENT Cultural Robustness & Diversity</strong> – Lot A Documentation</p>
    <p>Généré automatiquement depuis <code>LOT_A_CODE_DETAILLE.md</code></p>
</footer>
</body>
</html>"""
    
    # Écrire le fichier HTML
    Path(html_path).write_text(full_html, encoding="utf-8")
    
    size_kb = Path(html_path).stat().st_size / 1024
    print(f"✅ HTML créé : {html_path}")
    print(f"   Taille : {size_kb:.1f} Ko")
    print()
    print("📄 Pour créer le PDF :")
    print("   1. Ouvrez le fichier HTML dans votre navigateur")
    print("   2. Appuyez sur Ctrl+P (ou Cmd+P)")
    print("   3. Sélectionnez 'Enregistrer au format PDF'")


if __name__ == "__main__":
    convert_md_to_html(
        "docs/LOT_A_CODE_DETAILLE.md",
        "docs/LOT_A_CODE_DETAILLE.html"
    )



