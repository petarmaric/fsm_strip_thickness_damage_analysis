import argparse
import logging
import os
from timeit import default_timer as timer

import matplotlib
matplotlib.use('Agg') # Fixes weird segfaults, see http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server

from fsm_effective_stress import compute_damage, compute_effective_stress
from fsm_load_modal_composites import load_modal_composites
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from numpy.lib.recfunctions import append_fields


__version__ = '1.0.0'


FIGURE_SIZE = (11.7, 8.3) # In inches

DEFAULT_A = 1000.0

SUBPLOTS_SPEC = [
    {
        'line_plots': [ # 1st key is the ``main_key``
            # key, description
            ('D', ''),
        ],
        'ylabel': 'damage', # if ``None``, automatically determine from ``main_key``
    },
    {
        'line_plots': [
            ('sigma_cr',  'elastic'),
            ('sigma_eff', 'effective'),
        ],
        'ylabel': None,
    },
    {
        'line_plots': [
            ('m_dominant',    'elastic'),
            ('m_dominant_ve', 'viscoelastic'),
        ],
        'ylabel': 'dominant mode',
    },
]


def plot_marker(x, y):
    plt.scatter(
        x,
        y,
        s=18,
        marker='o',
        label="%g [mm]" % x,
        zorder=100,
    )
    plt.annotate(
        "%g" % y,
        xy=(x, y),
        xytext=(0, +4),
        textcoords='offset points',
    )

def plot_markers(markers, x, y):
    if not markers:
        return

    mask = np.nonzero(np.in1d(x, markers, assume_unique=True))
    for marker_x, marker_y in zip(x[mask], y[mask]):
        plot_marker(marker_x, marker_y)

def find_automatic_markers_on_mode_transitions(modal_composites, x, m_dominant_key):
    data = modal_composites[m_dominant_key]
    idx = np.nonzero(np.ediff1d(data, to_begin=0))
    return x[idx]

def plot_modal_composite(modal_composites, column_units, column_descriptions, markers=None, add_automatic_markers=False):
    logging.info("Plotting modal composites...")
    start = timer()

    def _get_column_title(column_name):
        description = column_descriptions[column_name]
        unit = column_units[column_name]
        return description if not unit else "%s [%s]" % (description, unit)

    a = modal_composites['a'][0]
    plt.suptitle("modal composites, strip length %f [mm]" % a)

    x = modal_composites['t_b']
    min_x = np.min(x)
    max_x = np.max(x)

    markers = markers or []
    if add_automatic_markers:
        for m_dominant_key in ['m_dominant', 'm_dominant_ve']:
            markers.extend(find_automatic_markers_on_mode_transitions(modal_composites, x, m_dominant_key))

    for ax_idx, spec in enumerate(SUBPLOTS_SPEC, start=1):
        plt.subplot(2, 2, ax_idx)

        main_key = spec['line_plots'][0][0] # 1st key is the ``main_key``
        for zorder, (key, description) in enumerate(spec['line_plots'], start=1):
            plt.plot(x, modal_composites[key], label=description, zorder=-zorder)
            plot_markers(markers, x, modal_composites[key])

        plt.xlim(min_x, max_x)

        plt.xlabel(_get_column_title('t_b'))
        plt.ylabel(spec['ylabel'] or _get_column_title(main_key))
        plt.legend()

    logging.info("Plotting completed in %f second(s)", timer() - start)

def dynamic_load_modal_composites(model_file, search_buffer=10**-10, **filters):
    modal_composites, column_units, column_descriptions = load_modal_composites(model_file, **filters)

    if modal_composites.size != 0:
        return modal_composites, column_units, column_descriptions

    a = filters.pop('a_fix')
    filters.update({
        'a_min': a - search_buffer,
        'a_max': a + search_buffer,
    })

    logging.warn("Could not find the exact value of a requested, expanding search condition to %(a_min)s <= a <= %(a_max)s", filters)
    return load_modal_composites(model_file, **filters)

def analyze_models(viscoelastic_model_file, elastic_model_file, report_file, markers=None, add_automatic_markers=False, **filters):
    with PdfPages(report_file) as pdf:
        elastic, column_units, column_descriptions = dynamic_load_modal_composites(elastic_model_file, **filters)
        viscoelastic, _, _ = dynamic_load_modal_composites(viscoelastic_model_file, **filters)

        omega = elastic['omega']
        omega_d = viscoelastic['omega']
        sigma_d = viscoelastic['sigma_cr']
        D = compute_damage(omega, omega_d)
        sigma_eff = compute_effective_stress(omega, omega_d, sigma_d)

        m_dominant_ve = viscoelastic['m_dominant']
        modal_composites = append_fields(
            elastic,
            names=['m_dominant_ve', 'D', 'sigma_eff'],
            data=[m_dominant_ve, D, sigma_eff],
            usemask=False
        )

        plot_modal_composite(modal_composites, column_units, column_descriptions, markers, add_automatic_markers)

        pdf.savefig()
        plt.close() # Prevents memory leaks

def configure_matplotlib():
    matplotlib.rc('figure',
        figsize=FIGURE_SIZE,
        titlesize='xx-large'
    )

    matplotlib.rc('figure.subplot',
        left   = 0.07, # the left side of the subplots of the figure
        right  = 0.98, # the right side of the subplots of the figure
        bottom = 0.06, # the bottom of the subplots of the figure
        top    = 0.91, # the top of the subplots of the figure
        wspace = 0.16, # the amount of width reserved for blank space between subplots
        hspace = 0.20, # the amount of height reserved for white space between subplots
    )

    matplotlib.rc('legend',
        fontsize='small',
    )

def main():
    # Setup command line option parser
    parser = argparse.ArgumentParser(
        description='Strip thickness-dependent damage analysis and visualization '\
                    'of the parametric model of buckling and free vibration '\
                    'in prismatic shell structures, as computed by the '\
                    'fsm_eigenvalue project.',
    )
    parser.add_argument(
        'viscoelastic_model_file',
        help="File storing the computed viscoelastic parametric model"
    )
    parser.add_argument(
        '-e',
        '--elastic_model_file',
        metavar='FILENAME',
        help="File storing the computed elastic parametric model, determined from '<viscoelastic_model_file>' by default"
    )
    parser.add_argument(
        '-r',
        '--report_file',
        metavar='FILENAME',
        help="Store the analysis report to the selected FILENAME, uses '<viscoelastic_model_file>.pdf' by default"
    )
    parser.add_argument(
        '--t_b-min',
        metavar='VAL',
        type=float,
        help='If specified, clip the minimum base strip thickness [mm] to VAL'
    )
    parser.add_argument(
        '--t_b-max',
        metavar='VAL',
        type=float,
        help='If specified, clip the maximum base strip thickness [mm] to VAL'
    )
    parser.add_argument(
        '--a',
        metavar='VAL',
        type=float,
        default=DEFAULT_A,
        help="Plot figures by fixing the selected strip length [mm] to VAL, %f by default" % DEFAULT_A
    )
    parser.add_argument(
        '--markers',
        metavar='POS',
        nargs='*',
        type=float,
        help='Plot marker(s) at specified strip thickness(es) [mm]'
    )
    parser.add_argument(
        '--add-automatic-markers',
        action='store_true',
        help='Plot automatic marker(s) on mode transitions'
    )
    parser.add_argument(
        '-q',
        '--quiet',
        action='store_const',
        const=logging.WARN,
        dest='verbosity',
        help='Be quiet, show only warnings and errors'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_const',
        const=logging.DEBUG,
        dest='verbosity',
        help='Be very verbose, show debug information'
    )
    parser.add_argument(
        '--version',
        action='version',
        version="%(prog)s " + __version__
    )
    args = parser.parse_args()

    # Configure logging
    log_level = args.verbosity or logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")

    configure_matplotlib()

    if not args.elastic_model_file:
        args.elastic_model_file = args.viscoelastic_model_file.replace('viscoelastic', 'elastic')

    if not args.report_file:
        args.report_file = os.path.splitext(args.viscoelastic_model_file)[0] + '.pdf'

    analyze_models(
        viscoelastic_model_file=args.viscoelastic_model_file,
        elastic_model_file=args.elastic_model_file,
        report_file=args.report_file,
        t_b_min=args.t_b_min,
        t_b_max=args.t_b_max,
        a_fix=args.a,
        markers=args.markers,
        add_automatic_markers=args.add_automatic_markers,
    )

if __name__ == '__main__':
    main()
