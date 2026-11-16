param([string]$Raiz = (Split-Path -Parent $PSScriptRoot))
$base = Join-Path $Raiz 'results\esquemas_presentacion'
$salida = Join-Path $base 'renderizados'
New-Item -ItemType Directory -Force -Path $salida | Out-Null
$fuentes = @(
    @{Archivo=(Join-Path $base 'fuente_original.pptx'); Diapositivas=@(3,4,5,6,7,8,9,10,13,14); Prefijo='diapositiva_'},
    @{Archivo=(Join-Path $base 'esquemas_para_jbook.pptx'); Diapositivas=@(15,16); Prefijo='diapositiva_'},
    @{Archivo=(Join-Path $base 'ampliados\esquemas_ampliados.pptx'); Diapositivas=(1..12); Prefijo='ampliado_'}
)
$aplicacion = New-Object -ComObject PowerPoint.Application
try {
    foreach($fuente in $fuentes) {
        $presentacion = $null
        try {
            $presentacion=$aplicacion.Presentations.Open($fuente.Archivo,-1,0,0)
            foreach($numero in $fuente.Diapositivas) {
                $archivo=Join-Path $salida ('{0}{1:D2}.png' -f $fuente.Prefijo,$numero)
                $presentacion.Slides.Item($numero).Export($archivo,'PNG',3200,1800)
            }
        } finally {
            if($null -ne $presentacion) {
                $presentacion.Close()
                [void][Runtime.InteropServices.Marshal]::ReleaseComObject($presentacion)
            }
        }
    }
} finally {
    if($aplicacion.Presentations.Count -eq 0){$aplicacion.Quit()}
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($aplicacion)
}
