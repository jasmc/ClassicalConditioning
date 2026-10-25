"""Compatible offline Blob download behavior; payload and notebook IDs unchanged."""
HANDLER = '<script id="embedded-source-downloads">\n// Blob downloads work in the in-app browser while retaining offline payloads.\nfor (const source of document.querySelectorAll(\'a[download][href^="data:"]\')) {\n  source.addEventListener(\'click\', event => {\n    event.preventDefault();\n    const raw=source.getAttribute(\'href\'), comma=raw.indexOf(\',\'), header=raw.slice(5,comma);\n    const content=raw.slice(comma+1), mime=header.split(\';\')[0] || \'application/octet-stream\';\n    const bytes=header.includes(\';base64\') ? Uint8Array.from(atob(content), c=>c.charCodeAt(0)) : new TextEncoder().encode(decodeURIComponent(content));\n    const url=URL.createObjectURL(new Blob([bytes],{type:mime}));\n    const download=document.createElement(\'a\'); download.href=url; download.download=source.download;\n    document.body.append(download); download.click(); download.remove();\n    setTimeout(()=>URL.revokeObjectURL(url),3000);\n  });\n}\n</script>'


def with_embedded_downloads(document):
    if 'id="embedded-source-downloads"' in document:
        return document
    if '</body>' not in document:
        raise ValueError('Guide document has no closing body')
    return document.replace('</body>', HANDLER + '</body>', 1)
