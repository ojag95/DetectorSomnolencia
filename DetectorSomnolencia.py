#Imports
from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import time
import logging as log
from openvino.inference_engine import IENetwork, IEPlugin

#Argumentos
def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-i", "--input",
                      help="Required. Path to video file or image. 'cam' for capturing video stream from camera",
                      required=True, type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with the "
                           "kernels implementations.", type=str, default=None)
    args.add_argument("-pp", "--plugin_dir", help="Optional. Path to a plugin folder", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. The demo will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Path to labels mapping file", default=None, type=str)
    args.add_argument("-pt", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.5, type=float)

    return parser

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    #Obtiene los argumentos de entrada
    args = build_argparser().parse_args()
    #Obitnen el modelo xml y posteriomente agrega la extension bi
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    log.info('Ruta del modelo '+model_bin)
    log.info('Ruta del xml '+model_xml)
    #Obtiene el dispositivo que se va a usar
    log.info("Inicializando plugin para dispositivo {} ...".format(args.device))
    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
    if args.cpu_extension and 'CPU' in args.device:
        plugin.add_cpu_extension(args.cpu_extension)
    # Cargando IR
    log.info("Leyendo IR...")
    net = IENetwork(model=model_xml, weights=model_bin)
    if plugin.device == "CPU":
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Las siguientes capas no son soportadas por el plugin para el dispositivo especificado {}:\n {}".
                      format(plugin.device, ', '.join(not_supported_layers)))
            log.error("Por favor intente especificar la ruta de una libreria de extensiones del cpu en los parametros usando -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
    assert len(net.inputs.keys()) == 1, "El demo soporta solo un tipo de topologia de entrada"
    assert len(net.outputs) == 1, "El demo soporta solo un tipo de topologia de salida"
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    log.info("Cargando IR al plugin...")
    exec_net = plugin.load(network=net, num_requests=2)
    n, c, h, w = net.inputs[input_blob].shape
    del net
    if args.input == 'cam':
        input_stream = 0
        log.info('La entrada seleccionada es la camara')
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "El archivo especificado no existe"
    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None
    #Inicia el proceso de captura de la entrada especificada
    cap = cv2.VideoCapture(input_stream)
    
    cur_request_id = 0
    next_request_id = 1

    log.info("Iniciando inferencia en modo asincrono...")
    log.info("Para cambiar entre modo asincrono y sincrono presione la tecla Tab")
    log.info("Para detener la ejecucion del demo presione la tecla Esc")
    is_async_mode = True
    render_time = 0
    #Obtiene los cuadros (frames) de la captura
    ret, frame = cap.read()
    print("Para cerrar la aplicación, presione 'CTRL+C' o alguna tecla con foco en la ventana de previsualización")
    #Inicia un ciclo que lee cada frame
    while cap.isOpened():
        if is_async_mode:
            ret, next_frame = cap.read()
        else:
            ret, frame = cap.read()
        if not ret:
            break
        initial_w = cap.get(3)
        initial_h = cap.get(4)
        #lOGICA DEL ALGORITMO

        #
        render_start = time.time()
        #Muestra en pantalla el cuadro con el frame actual
        cv2.imshow("Resultados de la deteccion", frame)
        render_end = time.time()
        render_time = render_end - render_start

        if is_async_mode:
            cur_request_id, next_request_id = next_request_id, cur_request_id
            frame = next_frame

        key = cv2.waitKey(1)
        if key == 27:
            break
        if (9 == key):
            is_async_mode = not is_async_mode
            log.info("Switched to {} mode".format("async" if is_async_mode else "sync"))

    cv2.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main() or 0)