import os
import argparse
from territory_finder import TerritoryFinder

parser = argparse.ArgumentParser(description="Find a territory for a trade outlet")

# Parse arguments
parser.add_argument("-c", "--coordinates", type=str, action='store',
    help="Please specify the file with coordinates", required=True)
parser.add_argument("-r", "--report", type=str, action='store',
    help="Please specify the 'Territory Management Report' file", required=True)
parser.add_argument("-o", "--output", type=str, action='store',
    help="Please specify the file you wish to load weights from(for example saved.h5)", required=False)
# parser.add_argument("-s", "--save", type=str, action='store', help="Specify folder to render simulation of network in", required=False)
# parser.add_argument("-x", "--statistics", action='store_true', help="Specify to calculate statistics of network(such as average score on game)", required=False)
# parser.add_argument("-v", "--view", action='store_true', help="Display the network playing a game of space-invaders. Is overriden by the -s command", required=False)

args = parser.parse_args()
# print(args.output)
# print(f"{args.coordinates}, {args.report}, {new_report}")

# set an output file name
new_report = os.path.splitext(args.report)[0] + " Updated.xlsx" if args.output == None else args.output
tf = TerritoryFinder(args.coordinates, args.report, new_report)

total_steps = 5

tf.log.info(f"Step 1 of {total_steps}: Loading and prepare data")
tf.load_data()

tf.log.info(f"Step 2 of {total_steps}: Validate the model")
tf.validate()

tf.log.info(f"Step 3 of {total_steps}: Train the model")
tf.fit()

tf.log.info(f"Step 4 of {total_steps}: Prepare report")
tf.get_report()

tf.log.info(f"Step 5 of {total_steps}: Save report")
tf.save_report()