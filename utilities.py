import numpy as np
import pandas as pd
import os, sys
import logging

def set_up_logging():
    """ Set up logging
    
    """
    os.makedirs('logs', exist_ok=True)
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d - %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    fh = logging.FileHandler(u"./logs/territory_finder.log", "w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(sh)
    log.addHandler(fh)
    
    return log

def align_value(value):
    """
    Избавится от крайних символов и дублирующихся запятых.
    Если получен список, то отсортировать по возрастанию

"""
    try:
        aligned = value
        # Избавится от крайних символов и дублирующихся запятых
        try:
            aligned = str(int(float(aligned)))
        except ValueError:
            aligned = aligned.strip().replace(', ,',',').replace(',  ,',',') \
                .replace(',,',',').replace(',,',',').replace(',,',',')
            while aligned[0] not in ['0','1','2','3','4','5','6','7','8','9']:
                aligned = aligned[1:]
            while aligned[-1] not in ['0','1','2','3','4','5','6','7','8','9']:
                aligned = aligned[:-1]
            aligned = np.array(aligned.split(',')).astype('float').astype('int')
            aligned = np.sort(aligned)
            aligned = ','.join(aligned.astype(str))
    except Exception as e:
        print(f"Возникла проблема при обработке кода {value}. Ошибка {e}")
        return value
    finally:
        return aligned
            
# def align_ship_to(value):
#     """Избавится от крайних символов и дублирующихся запятых
    
#     """
#     try:
#         # Избавится от крайних символов и дублирующихся запятых
#         aligned, value = str(value), str(value)
#         value = value.replace(',,',',').replace(',,',',').replace(',,',',') \
#             .replace(', ,',',').replace(',  ,',',').strip()
#         # remove non-digit symbols from left
#         while value[0] not in ['0','1','2','3','4','5','6','7','8','9']:
#             value = value[1:]
#         # remove non-digit symbols from right
#         while value[-1] not in ['0','1','2','3','4','5','6','7','8','9']:
#             value = value[:-1]
#         aligned = np.array(value.split(',')).astype('int')
#         aligned = np.sort(aligned)
#         aligned = ','.join(aligned.astype(str))
#     except Exception as e:
#         print(f"Возникла проблема при обработке кода {value}. Ошибка {e}")
#         return value
#     finally:
#         return aligned