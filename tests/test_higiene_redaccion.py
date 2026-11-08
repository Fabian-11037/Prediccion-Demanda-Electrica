"""La entrega no debe mencionar otros proyectos del autor; permite bibliografía externa."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PROHIBIDAS = ['Time-Series-Project', 'proyecto previo', 'proyecto anterior']
EXENTOS = {'CONTEXTO_PROYECTO.md', 'CLAUDE.md', 'AGENTS.md', 'test_higiene_redaccion.py'}
EXT = {'.md', '.py', '.json', '.txt', '.csv', '.ipynb', '.yml', '.yaml', '.toml'}


def referencias_prohibidas(texto):
    return [s for s in PROHIBIDAS if s.casefold() in texto.casefold()]


def archivos_de_entrega():
    # No analizar dependencias, cachés, HTML compilado ni instantáneas de terceros.
    for carpeta in ['book', 'scripts', 'src', 'literature']:
        for p in (RAIZ / carpeta).rglob('*'):
            if any(x in {'_build', '_static', '__pycache__'} for x in p.parts):
                continue
            if p.is_file() and p.suffix in EXT and p.name not in EXENTOS:
                yield p
    yield RAIZ / 'README.md'


def test_sin_referencias_a_trabajos_previos():
    malos=[]
    for p in archivos_de_entrega():
        texto=p.read_text(encoding='utf-8',errors='strict')
        malos.extend((str(p.relative_to(RAIZ)),s) for s in referencias_prohibidas(texto))
    assert not malos,malos


def test_permite_publicaciones_de_otros_autores():
    assert not referencias_prohibidas('Liu et al. presentan Time-MMD; Bi et al. presentan NSW-EPNews para predicción de precio eléctrico.')


def test_detecta_referencia_al_repositorio_del_autor():
    assert referencias_prohibidas('Repositorio Time-Series-'+'Project')
