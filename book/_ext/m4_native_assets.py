"""Aísla los recursos del panel; se cargan dentro de su iframe, no en el libro."""
def isolate(app,config):
    def keep(item):
        name=item[0] if isinstance(item,(tuple,list)) else item
        return not str(name).startswith('inferencia_m4/')
    config.html_js_files=[item for item in config.html_js_files if keep(item)]
    config.html_css_files=[item for item in config.html_css_files if keep(item)]

def setup(app):
    app.connect('config-inited',isolate,priority=900)
    return {'version':'1.0','parallel_read_safe':True,'parallel_write_safe':True}
