"""GROBID client: PDF -> TEI XML -> structured data."""

from __future__ import annotations

import os
from typing import Any

import aiohttp
from lxml import etree

GROBID_URL = os.environ.get("GROBID_URL", "http://localhost:8070")

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def _text(el) -> str:
    """Get all text content from an lxml element, stripping whitespace."""
    if el is None:
        return ""
    return " ".join(el.itertext()).strip()


def parse_tei_xml(xml_str: str) -> dict[str, Any]:
    """Extract structured data from GROBID TEI XML output."""
    root = etree.fromstring(xml_str.encode("utf-8") if isinstance(xml_str, str) else xml_str)

    # Title
    title_el = root.find(".//tei:titleStmt/tei:title", TEI_NS)
    title = _text(title_el)

    # Abstract
    abstract_el = root.find(".//tei:profileDesc/tei:abstract", TEI_NS)
    abstract = _text(abstract_el)

    # Authors
    authors = []
    for author_el in root.findall(".//tei:fileDesc/tei:sourceDesc//tei:author", TEI_NS):
        persname = author_el.find("tei:persName", TEI_NS)
        if persname is not None:
            forename = _text(persname.find("tei:forename", TEI_NS))
            surname = _text(persname.find("tei:surname", TEI_NS))
            name = f"{forename} {surname}".strip()
            if name:
                authors.append({"name": name})

    # Sections
    sections = []
    for div in root.findall(".//tei:body/tei:div", TEI_NS):
        head = div.find("tei:head", TEI_NS)
        heading = _text(head) if head is not None else ""
        paragraphs = [_text(p) for p in div.findall("tei:p", TEI_NS)]
        body_text = "\n".join(p for p in paragraphs if p)
        if heading or body_text:
            sections.append({"heading": heading, "text": body_text})

    # References
    references = []
    for bib in root.findall(".//tei:listBibl/tei:biblStruct", TEI_NS):
        ref_title_el = bib.find(".//tei:title[@level='a']", TEI_NS)
        ref_title = _text(ref_title_el)
        ref_authors = []
        for a in bib.findall(".//tei:author/tei:persName", TEI_NS):
            fn = _text(a.find("tei:forename", TEI_NS))
            sn = _text(a.find("tei:surname", TEI_NS))
            n = f"{fn} {sn}".strip()
            if n:
                ref_authors.append(n)
        ref_year_el = bib.find(".//tei:date[@type='published']", TEI_NS)
        ref_year = ref_year_el.get("when", "") if ref_year_el is not None else ""
        doi_el = bib.find(".//tei:idno[@type='DOI']", TEI_NS)
        ref_doi = _text(doi_el)
        if ref_title:
            references.append({
                "title": ref_title,
                "authors": ref_authors,
                "year": ref_year[:4] if ref_year else None,
                "doi": ref_doi or None,
            })

    # Keywords
    keywords = []
    for kw in root.findall(".//tei:profileDesc/tei:textClass/tei:keywords/tei:term", TEI_NS):
        t = _text(kw)
        if t:
            keywords.append(t)

    return {
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "sections": sections,
        "references": references,
        "keywords": keywords,
    }


async def process_pdf_from_url(pdf_url: str) -> dict[str, Any]:
    """Download a PDF from a URL and process it through GROBID.

    Returns parsed TEI data dict with title, abstract, authors, sections,
    references, and keywords.
    """
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Download the PDF
        async with session.get(pdf_url) as resp:
            if resp.status != 200:
                raise ValueError(f"Failed to download PDF: HTTP {resp.status}")
            pdf_bytes = await resp.read()

        # Send to GROBID
        grobid_endpoint = f"{GROBID_URL}/api/processFulltextDocument"
        form = aiohttp.FormData()
        form.add_field(
            "input",
            pdf_bytes,
            filename="paper.pdf",
            content_type="application/pdf",
        )

        async with session.post(grobid_endpoint, data=form) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise ValueError(f"GROBID processing failed: HTTP {resp.status}: {error[:200]}")
            tei_xml = await resp.text()

    return parse_tei_xml(tei_xml)
