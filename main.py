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
new_report = os.path.splitext(args.report)[0] + " Updated" if args.output == None else args.output
territory_finder = TerritoryFinder(args.coordinates, args.report, new_report)

total_steps = 9

territory_finder.log.info(f"Step 1 of {total_steps}: Loading and prepare data")
territory_finder.load_data()

territory_finder.log.info(f"Step 2 of {total_steps}: Validate the model")
territory_finder.validate()

territory_finder.log.info(f"Step 3 of {total_steps}: Train the model")
territory_finder.fit()

territory_finder.log.info(f"Step 4 of {total_steps}: Prepare report")
# territory_finder.get_report()

territory_finder.log.info(f"Step 5 of {total_steps}: Save report")
# territory_finder.save_report()

# print('Step 7 of 9: Define top 3 classes')

# print("last line")