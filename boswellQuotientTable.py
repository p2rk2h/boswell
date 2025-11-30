import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from LetterGrade import LetterGrade     #####

# utility functions:	#####################

import os , sys
import math
from typing import Dict, List, Any, Tuple
from pathlib import Path
from IPython.display import display

def is_jupyter() -> bool :
    try:
        ipy = get_ipython()
        if ( ipy is None ) :  # Not in an IPython environment
            return False
        return 'IPKernelApp' in ipy.config
    except (NameError, ImportError, AttributeError):
        return False

def running_environment():
    if is_jupyter():
        return "Jupyter Notebook/Lab"
    else:
        try:
            from IPython import get_ipython
            ipy = get_ipython()
            return "IPython Shell"
        except (NameError, ImportError):
            return "Terminal/Script"

def clean_terminal_output(df):
    df.columns = [c.replace('<br>', ' ') for c in df.columns]
    df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: x.replace('\n', ' ') if isinstance(x, str) else x)
    return df

def printDfrows(df: pd.DataFrame, max_rows: int | None = 100) -> None:
    with pd.option_context('display.max_rows', max_rows or df.shape[0],
                           'display.max_columns', None,
                           'display.max_colwidth', None,
                           'display.width', None):
        print(f"df ({df.shape[0]}×{df.shape[1]}):\n{df}\n{'*' * 50}\n")

def exportPng( dfStyle , dir , pngName ) :
    try:
        output_path = os.path.join( dir , pngName )
        dfi.export( dfStyle , output_path, table_conversion='selenium')
        print("Successfully exported " + pngName )
    except Exception as e:
        print( dir + pngName , "#############" )	#####
        print(f"Export error: {e}")

def export_and_display_table(
    styled_df,
    original_df,
    save_table_images=False,
    output_dir="",
    output_filename="BoswellQuotientTableOutput.png"
):
    """
    Exports a styled DataFrame to PNG (if requested) and displays it appropriately
    based on the runtime environment (Jupyter, IPython, or terminal).

    Args:
        styled_df: Styled DataFrame (e.g., `df.style`).
        original_df: Original DataFrame (for terminal/IPython output).
        save_table_images: If True, exports the table as a PNG.
        output_dir: Directory to save the PNG.
        output_filename: Filename for the exported PNG.
    """
    # Export to PNG if requested
    if save_table_images:
        exportPng( styled_df , output_dir , output_filename )

    # Display or print based on environment
    if is_jupyter():
        display(styled_df)
    else:
        cleaned_df = clean_terminal_output(original_df)
        env = running_environment()
        print(f"Running environment: {env}")
        printDfrows(cleaned_df)  # For terminal/IPython shell

#####	import dataframe_image, a PIPy package
#####	if missing, print install instuctions (w/ selenium package)
#####	so calculated dataframes can be exported as png/jpg files
try :
    import dataframe_image as dfi
    saveTableImages = True
    print( "save table images :\t" , saveTableImages )
except Exception as e:
    print( f"import dfi failure  error: {e}" )
    saveTableImages = False
    print( "Saving table output to png/jpg disabled" )
    print( "dataframe_image installation instruction:" )
    print( "\t1) best to create a new virtual environment for a python version >= 3.10" )
    print( "\t2) activate the newly created environment and install the following packages:" )
    print( "\t2) 'pandas numpy matplotlib ipython seaborn ipython jupyter'" )
    print( "\t3) install pip packages 'pip install dataframe_image selenium'" )
    print( "\t4) relaunch jupyter notebook or rerun python scripts\n" )

# Type definition for clarity
ModelData = Dict[str, List[Dict[str, Any]]]

def _parse_table_line(line: str) -> List[str]:
    """Helper to parse a standard Markdown table row (excluding header/separators)."""
    # Split by '|', strip whitespace, and filter out empty strings (from leading/trailing '|')
    return [part.strip() for part in line.split('|') if part.strip()]

def calculate_latency_report(report_dir: str) -> ModelData:
    """
    Reads the timing_report.md file, extracts total time, model timings, 
    calculates maximum times, and computes a 'latency' score for each model entry.

    The latency calculation is based on an assumed formula: 
    Latency = 100 - Efficiency, where Efficiency is a weighted mean of 
    normalized essay time and normalized grading time.
    
    Args:
        report_dir: The directory path where timing_report.md is located.

    Returns:
        A dictionary mapping base model names to a list of timing results.
    """
    timing_filepath = os.path.join(report_dir, 'timing_report.md')
    model_data: ModelData = {}
    total_time_ms: float = 0.0
    max_essay_time: float = 0.0
    max_grading_time: float = 0.0
    
    essay_entries: List[Tuple[str, float]] = [] # Store temporary essay data for later pairing
    
    # State flags to control parsing flow
    parsing_essay_table = False
    parsing_grading_table = False
    
    try:
        with open(timing_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # --- 1. Extract Total Time ---
                if any(s in line for s in ['**Essay generation', '**Grading']):
                    try:
                        # Assumes format: "...: 12.34%"
                        parts = [part.strip() for part in line.split(':') if part.strip()]
                        # Extract the number before the last character (which is '%')
                        time_percentage = float(parts[-1][:-1])
                        total_time_ms += time_percentage
                        print(f"Total time found: {total_time_ms}, parts: {parts}")  ###
                    except (IndexError, ValueError) as e:
                        print(f"WARNING: Failed to parse total time line: '{line}'. Error: {e}")
                        
                # --- 2. Start Essay Timing Table Parsing ---
                elif '## Essay Generation Timing' in line:
                    parsing_essay_table = True
                    # Skip the next 3 lines (e.g., header, separator, first data row)
                    for _ in range(3):
                        next(f, None)
                    continue

                # --- 3. Start Total/Grade Timing Table Parsing ---
                elif '## Grading Timing' in line: # Assuming this is the header for the second table based on original logic
                    parsing_essay_table = False
                    parsing_grading_table = True
                    # Skip the next 3 lines (e.g., header, separator, first data row)
                    for _ in range(3):
                        next(f, None)
                    continue

                # --- 4. Process Essay Timing Table Rows ---
                if parsing_essay_table and line and not line.startswith('|---|'):
                    if '**' in line : # Stop parsing if another section starts
                        parsing_essay_table = False
                        continue
                    
                    parts = _parse_table_line(line)
                    if len(parts) >= 2:
                        try:
                            # Model Full Name is parts[0], Base Model Name is split by ':'
                            model_full = parts[0]
                            base_model = model_full.split(':', 1)[0]
                            essay_time = float(parts[1])
                            
                            max_essay_time = max(max_essay_time, essay_time)
                            
                            # Store the essay time data
                            if base_model not in model_data:
                                model_data[base_model] = []
                            
                            # Store data structure used in original code
                            model_data[base_model].append(
                                {'m': model_full, 't': {'essayT': essay_time}}
                            )
                            
                        except (IndexError, ValueError) as e:
                            print(f"WARNING: Failed to parse essay table row: '{line}'. Error: {e}")
                    
                # --- 5. Process Total/Grade Timing Table Rows ---
                elif parsing_grading_table and line and not line.startswith('|---|'):
                    if not line: # Blank line ends the section
                        parsing_grading_table = False
                        break
                        
                    parts = _parse_table_line(line)
                    if len(parts) >= 3: # Assuming columns are Model | Total T | Grade T
                        try:
                            model_full = parts[0]
                            base_model = model_full.split(':', 1)[0]
                            # total_time = float(parts[1]) # total_time is not used in latency formula
                            grade_time = float(parts[2])
                            
                            max_grading_time = max(max_grading_time, grade_time)
                            
                            # Find the matching entry in model_data and update it
                            if base_model in model_data:
                                found = False
                                for entry in model_data[base_model]:
                                    if entry['m'] == model_full:
                                        entry['t']['gradeT'] = grade_time
                                        found = True
                                        
                                        # Calculate Latency Score
                                        normalized_essay = entry['t']['essayT'] / max_essay_time if max_essay_time else 1.0
                                        normalized_grade = grade_time / max_grading_time if max_grading_time else 1.0
                                        
                                        # Original formula: 100 - int(50 * (2 - (norm_essay + norm_grade)) + 0.499999)
                                        # This simplifies to: 100 - round(50 * (2 - (norm_essay + norm_grade)))
                                        efficiency_score = 50 * (2 - (normalized_essay + normalized_grade))
                                        
                                        # Using standard Python rounding
                                        latency = 100 - round(efficiency_score) 
                                        
                                        entry['latency'] = latency
                                        break
                                
                                if not found:
                                    print(f"WARNING: Found grade time for unmatched entry: {model_full}")
                                    
                        except (IndexError, ValueError) as e:
                            print(f"WARNING: Failed to parse grade table row: '{line}'. Error: {e}")
                            
    except FileNotFoundError:
        print(f"ERROR: Timing report file not found at {timing_filepath}")
        return {}
        
    print(f"\n--- Summary ---")
    print(f"Total Time Found: {total_time_ms:.2f}")
    print(f"Max Essay Time (emxt): {max_essay_time:.4f}")
    print(f"Max Grading Time (gmxt): {max_grading_time:.4f}")

    return model_data

#   =====

from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from statistics import median

# These must be explicitly imported — no magic globals
# from latency_report import calculate_latency_report
# from your_module import LetterGrade  # ← MUST DEFINE THIS
#  https://x.com/i/grok?conversation=1992040135589011577

def get_dataframe_with_latency(report_dir: str) -> pd.DataFrame:
    """
    Loads grade data from 'grades_table.csv', integrates latency scores, 
    and appends statistical summary rows (Medians, Bias Corrections, Latency).

    Args:
        report_dir: The directory path where grades_table.csv and the timing report
                    are located.

    Returns:
        The processed pandas DataFrame containing grade data and summary rows.
    """
    
    # 1. Get latency data
    model_latency_data = calculate_latency_report(report_dir)

    # 2. Load grades table
    grades_filepath = os.path.join(report_dir, "grades_table.csv")
    df = pd.read_csv(grades_filepath, encoding='utf-8')
    
    # Clean whitespace
    df.columns = [c.strip() for c in df.columns]
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Identify model columns: everything except first (ID) and last two (metadata + percentage)
    full_model_columns = df.columns[1:-2].tolist()   # keep original full names for matching

    # 3. Build latency column values (header + one value per model row)
    latency_col: List[Any] = ['latency']  # header

    for col_full in full_model_columns:
        base_name = col_full.split(':', 1)[0]
        score = float('nan')
        
        if base_name in model_latency_data:
            for entry in model_latency_data[base_name]:
                if entry['m'] == col_full:
                    score = entry.get('latency', float('nan'))
                    break
        else:
            print(f"WARNING: No latency data for {base_name}")
            
        latency_col.append(score)

    # 4. Simplify column names NOW (after latency matching)
    df.columns = [col.split(':', 1)[0] for col in df.columns]

    # Replace --- / nan with 0
    df = df.replace(['---', 'nan', float('nan')], '0')

    # Drop the old percentage column (it's the last one)
    if df.columns[-1].lower() == 'percentage':
        df = df.iloc[:, :-1]   # remove last column

    # === INSERT LATENCY AS A REAL COLUMN (replaces Percentage) ===
    model_row_count = len(df)
    if len(latency_col) == model_row_count + 1:   # +1 because of header
        df['latency'] = latency_col[1:]           # skip the 'latency' header
    else:
        print(f"ERROR: Latency length mismatch: {len(latency_col)-1} vs {model_row_count}")
        df['latency'] = float('nan')

    # 5. Add column medians row
    median_row = ['col median/bias']
    grade_cols = [c for c in df.columns[1:]]  #  includes latency column
    
    for col in grade_cols:
        try:
            med = LetterGrade.median(df[col].tolist())
            median_row.append(med)
        except Exception as e:
            print(f"Median error on {col}: {e}")
            median_row.append('ERR')
    # Add median of latency too if you want (optional)
    # median_row.append(median([x for x in df['latency'] if pd.notna(x)] or [0]))

    df.loc[len(df)] = median_row
    
    return df

#   =====
'''
dir = '/Users/MingLuh/Downloads/botwell/results/20251030-171718-pol_sci_1_pd2/'
# dir = '/Users/MingLuh/Downloads/botwell/results/20250331-ai_policy/'

if ( dir[ -1 ] != '/' and dir[ -1 ].isalnum( ) ) :
    dir += '/'

print( "versions:\t" , sys.version.split( )[ 0 ] , pd.__version__ , dfi.__version__ )
print( "dir:\t\t" , dir )

df = get_dataframe_with_latency( dir )

nRow , nCol = df.shape

dfr = df.copy( )
model_names = dfr.iloc[:nRow, 0].tolist()
mxCol0Char = max( ( max( [ len( m ) for m in model_names ] ) + 2 ) // 3 + 1 , 18 )
dfr.iloc[:nRow, 0] = wrapLongString(model_names, max_len = mxCol0Char, wrapStr = '<br>' )
# print( dfr.iloc[ : nRow , 0 ].tolist( ) )

# float_cols = dfr.select_dtypes(include='float').columns
# print( float_cols , dfr['latency'] )
dfr['latency'] = dfr['latency'].apply(lambda x: int(round(x)) if not isinstance( x , str ) else x )
    
styler = dfr.style.set_properties(**{'text-align': 'left', 'white-space': 'pre'})  #  .format("{:d}",subset=float_cols)

export_and_display_table( styler, df, saveTableImages , output_dir = dir , output_filename = "RawGradeTableOutput.png" )
'''
#   =====

def getDfNormalizedGrades(df):
    '''
    Normalizes grades by adding bias correction (in LetterGrade numeric space)
    df: DataFrame with 2 extra rows: [col median/bias] and [bias corrections]
    Returns: df with normalized grades + new summary rows
    '''
    df = df.copy()
    
    # 0. Add bias corrections row
    model_row_count = len(df) - 1
    col_medians = df.iloc[-1, 1:-2].tolist()   # all column medians except first/label and last 2 columns
                                               # (median grade and latency)
    gradingBiasMdn = LetterGrade.median(col_medians)
    
    bias_corr = ['bias corrections']
    for med in col_medians:
        diff = float(LetterGrade(gradingBiasMdn) - LetterGrade(med))
        bias_corr.append(f"{diff:.3f}")
    # Optional: add overall bias and self-grading median at the end
    bias_corr.append(f"{gradingBiasMdn}")  
    selfGradingMdn = LetterGrade.median([df.iloc[i, i+1] for i in range(model_row_count)])
    bias_corr.append(f"{selfGradingMdn}")

    df.loc[len(df)] = bias_corr
    
    nRow2, nCol = df.shape
    nRow = nRow2 - 2                    # number of actual models/chatbots

    # === CORRECT: Get the global grading bias from the bias corrections row ===
    original_overall_median = df.iloc[ : nRow , nRow ].tolist()
    gradingBiasMdn = df.iloc[nRow + 1, nRow + 1]      # ← this is correct
    selfGradingMdn = df.iloc[ nRow + 1 , nRow + 2 ]
    
    # === 1. Apply bias correction to every grade ===
    for i in range(nRow):
        for j in range(1, nRow + 1):
            original_grade = df.iloc[i, j]
            bias_correction = float(df.iloc[nRow + 1, j])   # this is a float like "0.320"
            normalized = LetterGrade(original_grade) + bias_correction
            df.iloc[i, j] = str(normalized)                # store as string like "A-"

        # Compute normalized median grade for this model (row)
        row_grades = df.iloc[i, 1:nRow + 1].tolist()
        df.iloc[i, nRow + 1] = LetterGrade.median(row_grades)

    # === 2. Summary statistics ===
    
    normalized_overall_median = LetterGrade.median(df.iloc[:nRow, nRow + 1].tolist())

    df.rename(columns={df.columns[nRow + 1]: 'normalized median grade'}, inplace=True)

    # === 3. Add new row: column medians of normalized grades ===
    colMdnRow = ['column medians']
    for j in range(1, nRow + 1):
        col_values = df.iloc[:nRow, j].tolist()
        colMdnRow.append(LetterGrade.median(col_values))

    normalized_column_median = LetterGrade.median(colMdnRow[1:])  # median of all column medians
    colMdnRow.append(normalized_overall_median)                  # median of row medians
    colMdnRow.append(normalized_column_median)                  # overall fairness metric

    # Insert as new row (replaces old col median row or adds below), but not needed after normalization, 
    # since all model column medians are now the same as the grading bias median or df0.iloc[ -1 , -2 ] 
    # df.loc[nRow] = colMdnRow   #  only if to overwrite old "col median/bias" row (unneeded)

    # === Diagnostic print (very useful!) ===
    # print(f"Original median: {original_overall_median}")
    print(f"Applied global bias: {gradingBiasMdn}", df.iloc[ -2 , -2 ] )  #  -2th row due to biasCorr row
    print(f"New normalized median: {normalized_overall_median}")
    print(f"Normalized column median: {normalized_column_median}")

    return df
'''
dfn = getDfNormalizedGrades( df )

dfn
'''
#   =====

import math
from fractions import Fraction
from functools import reduce

def approximate_ratio_flexible( values , max_error = 0.01 ) :
    """
    Try to find a simple ratio that approximates the input within max_error.
    """
    # Normalize to sum = 1 (if it's a distribution)
    total = sum( values )
    if ( total == 0 ) :
        return ":".join( "0" for _ in values )
    norm = [ v / total for v in values ]
    
    # Try denominators from 1 to 100
    for denom in range( 1 , 101 ) :
        ints = [ round( p * denom ) for p in norm ]
        if ( sum( ints ) == 0 ) :
            continue
        # Rescale to actual values
        approx = [ i / denom for i in ints ]
        # Check error
        if ( all( abs( a - b ) <= max_error for a , b in zip( approx , norm ) ) ) :
            # Simplify ints
            g = reduce( math.gcd , [ x for x in ints if ( x != 0) ] )
            if ( g > 1 ) :
                ints = [ x // g for x in ints ]
            return ( ":".join( map( str , ints ) ) )
    # Fallback: use exact method
    return ( exact_ratio_string( values ) )

def exact_ratio_string( values ) :
    from fractions import Fraction
    from functools import reduce
    fracs = [ Fraction( v ).limit_denominator( 1000000 ) for v in values ]
    denoms = [ f.denominator for f in fracs ]
    lcm = reduce( lambda a , b : a * b // math.gcd( a , b ) , denoms )
    ints = [int( f.numerator * (lcm // f.denominator ) ) for f in fracs ]
    g = reduce( math.gcd , [ x for x in ints if ( x != 0 ) ] )
    if ( g > 0 ) :
        ints = [ x // g for x in ints ]
    return ( ":".join( map( str , ints ) ) )
'''
print( approximate_ratio_flexible( [ 0.5 , 0.2 , 0.2 , 0.1 ] ) , 
       approximate_ratio_flexible( [ 0.51 , 0.21 , 0.21 , 0.097 ] )
'''
#   =====

def map2gradeDct(values: list[float], grade_dict: dict[str, float]) -> list[str]:
    """
    Map numeric values to letter grades using thresholds.
    Assumes grade_dict is ordered: higher key = better grade.
    Example: {'D': 1.0, 'C': 2.0, 'B': 3.0, 'A': 3, 'A+': 4.25}
    """
    # Ensure dict is ordered and extract thresholds
    thresholds = list(grade_dict.items())  # [('D', 1.0), ('C', 2.0), ...]
    result = []

    for val in values:
        # Find the first (lowest) grade that val does NOT exceed
        for grade, threshold in thresholds:  # start from best
            if val < threshold:
                result.append(grade)
                break
        else:
            result.append(list(grade_dict.keys())[0])  # fallback: worst grade

    return result

def map2fiveStats(values: list[float], reverse: bool = False) -> list[str]:
    """
    Assign letter grades using five-number summary.
    reverse=True → higher original value = better (e.g. latency)
    """
    data = [-x for x in values] if reverse else values.copy()
    
    mn, q1, md, q3, mx = LetterGrade.five_number_summary(data)
    
    # Convert to numeric if needed
    to_num = lambda x: LetterGrade(x).numeric_value if isinstance(x, (str, LetterGrade)) else x
    mn, q1, md, q3, mx = map(to_num, [mn, q1, md, q3, mx])

    # Handle collapsed distributions gracefully
    if abs(mn - q1) < 1e-8:
        q1 = (mn + md) / 2
    if abs(q3 - mx) < 1e-8:
        q3 = (md + mx) / 2

    result = []
    for val in data:
        if abs(val - mn) < 1e-8:
            result.append('A+')
        elif val < q1:
            result.append('A')
        elif val < md:
            result.append('B')
        elif val < q3:
            result.append('C')
        else:
            result.append('D')  # top 25% get D? Or make it 'C-'?

    return result

def wrapLongString(strings: list[str], max_len: int = 18 , wrapStr = '\n' ) -> list[str]:
    """
    Wrap long strings at the last space before max_len.
    Safe for strings with no spaces.
    """
    result = []
    for s in strings:
        if len(s) <= max_len:
            result.append(s)
            continue
        
        # Find last space before max_len + buffer
        wrap_point = s.rfind(' ', 0, max_len + 5)
        if wrap_point == -1:
            # No space found → force break at max_len
            wrap_point = max_len
        
        wrapped = s[:wrap_point] + '\n' + s[wrap_point+1:] if wrap_point > 0 else s  #  use '<br>' for dataframe_image styling
        result.append(wrapped)
    
    return result
'''
v = [1.375, 4.125, 6.875, 9.625, 12.375, 15.125, 17.875, 20.625, 23.375, 26.125, 28.875]
k = ['A+', 'A', 'A-', 'A-/B+', 'B+', 'B', 'B-', 'B-/C+', 'C+', 'C', 'C-']
l = [11.25, 8.25, 5.0, 6.75, 7.25, 4.75, 4.5, 6.75, 5.5, 5.25, 4.25, 5.5]

map2gradeDct( l , dict(zip( k , v ) ) )
'''

#   =====

from statistics import median

def getDfBoswellQuotientTable(df2, df0, wgtDct=None):
    df = df2.copy()
    if wgtDct is None:
        wgtDct = {"normalized grades": 0.5, "accuracy": 0.2, "consistency": 0.2, "latency": 0.1}
    
    nRow2, nCol = df.shape
    nRow = nRow2 - 2  # original number of models

    # Grade scale setup
    A_plus = LetterGrade('A+')
    dacry = A_plus - LetterGrade('A')
    facry = dacry * nRow
    grades = list(LetterGrade.GRADE_POINTS.keys())
    thresholds = [facry * (i + 0.5) for i in range(len(grades))]
    dgry  = dict(zip(grades, thresholds))                    # for accuracy
    dgry2 = dict(zip(grades, [i * 0.5 for i in range(len(grades))]))  # for consistency
    # print( dacry , facry , grades , thresholds , dgry , dgry2 )  ###

    # === Extract key medians ===
    gradingBiasMdn = LetterGrade.median(df.iloc[nRow, 1:nRow+1].tolist())
    normalizedGradMed = LetterGrade.median(df.iloc[:nRow, nRow].tolist())
    original_grades = [g.strip() for g in df0.iloc[:nRow, nRow].tolist()]
    biasCorrMdn = LetterGrade.median( [ float( s ) for s in df.iloc[ nRow + 1 , 1 : nRow + 1 ].to_list( ) ] )

    # === Build metric rows ===
    rows = [ ]

    # 1–4: Original, bias, correction, normalized
    rows.append([ 'Boswell Quotient metrics' , 'weights' ] + df0.iloc[ : nRow , 0 ].to_list( ) + [ 'median' ])
    rows.append(['original grades', '0'] + original_grades + [LetterGrade.median(original_grades)])
    rows.append(['grading biases',  '0'] + df.iloc[nRow, 1:nRow+1].tolist() + [gradingBiasMdn])
    rows.append(['bias corrections','0'] + df.iloc[nRow+1, 1:nRow+1].tolist() + [f"{biasCorrMdn:.3f}"])
    rows.append(['normalized grades', f"{wgtDct['normalized grades']:.2f}"] + df.iloc[:nRow, nRow].tolist() \
                + [normalizedGradMed])
    # print( gradingBiasMdn , normalizedGradMed , biasCorrMdn , [ r for r in rows ])  ###
    
    # 5–6: Accuracy
    raw_accuracy = ['raw accuracy', '0']
    acc_grades   = ['accuracy grades', f"{wgtDct['accuracy']:.2f}"]
    for i in range(nRow):
        diffs = [A_plus - LetterGrade(df.iloc[i, j]) for j in range(1, nRow+1)]
        total_diff = sum(diffs)
        raw_accuracy.append(total_diff)
        acc_grades.append(map2gradeDct([total_diff], dgry)[0])
    raw_accuracy.append(median(raw_accuracy[2:]))
    acc_grades.append(LetterGrade.median(acc_grades[2:]))
    
    raw_accuracy = raw_accuracy[ : 2 ] + [ f"{f:.3f}" for f in raw_accuracy[ 2 : ] ]

    rows.append(raw_accuracy)
    rows.append(acc_grades)
    # print( 'raccuracy:' , raw_accuracy )  ###
    # print( 'accuracy1r:' , acc_grades )  ###

    # 7–8: Consistency (MAD from row median)
    mad_values = ['abs median deviation', '0']
    cons_grades = ['consistency', f"{wgtDct['consistency']:.2f}"]
    for i in range(nRow):
        row_median = LetterGrade(df.iloc[i, nRow + 1])
        mad = sum(abs(LetterGrade(df.iloc[i, j]) - row_median) for j in range(1, nRow+1))
        mad_values.append(mad)
        cons_grades.append(map2gradeDct([mad], dgry2)[0])
    mad_values.append(median(mad_values[2:]))
    cons_grades.append(LetterGrade.median(cons_grades[2:]))
    
    mad_values = mad_values[ : 2 ] + [ f"{f:.3f}" for f in mad_values[ 2 : ] ]
    
    rows.append(mad_values)
    rows.append(cons_grades)
    
    # print( "aconsens :" , mad_values )  ###
    # print( "consens :" , cons_grades )  ###

    # 9–10: Latency
    raw_latency = ['raw latency', '0'] + df.iloc[:nRow, -1].tolist()
    latency_vals = [float(x) for x in raw_latency[2:]]
    raw_latency.append(median(latency_vals))
    latency_grades = ['latency', f"{wgtDct['latency']:.2f}"] + map2fiveStats(latency_vals) ### , reverse=True)
    latency_grades.append(LetterGrade.median(latency_grades[2:]))
    
    raw_latency = raw_latency[ : 2 ] + [ f"{s:.0f}" for s in raw_latency[ 2 : ] ]
    
    rows.append(raw_latency)
    rows.append(latency_grades)
    # print( raw_latency )  ###
    # print( latency_grades )  ###

    # === Final Boswell Quotient ===
    final_row = ['Boswell Weighted Quotient', '0'] + [' '] * (nRow + 1)
    rows.append(final_row)

    # Build DataFrame
    bq = pd.DataFrame(rows).T
    
    # print( bq.shape , bq.columns , bq.iloc[ : , 0 ] , [ len( r ) for r in rows ] )  ###

    # Compute weighted composite
    weights = [0.0] + [float(bq.iloc[1, j]) for j in range(1, len(bq.columns)-1)] + [0.0]
    score_col = len(bq.columns) - 1

    for i in range(2, nRow + 2):
        weighted_sum = 0.0
        total_weight = 0.0
        for j in range(1, len(weights)-1):
            if weights[j] > 0:
                grade_val = LetterGrade(bq.iloc[i, j]).numeric_value
                weighted_sum += weights[j] * grade_val
                total_weight += weights[j]
        if total_weight > 0:
            bq.iloc[i, score_col] = str(LetterGrade.from_numeric(weighted_sum / total_weight))

    # Median of final scores
    final_scores = bq.iloc[2:nRow+2, score_col].to_list( )
    # print( final_scores )  ###
    bq.iloc[nRow + 2, score_col] = LetterGrade.median(final_scores)
    bq.iloc[ 1 , score_col ] = approximate_ratio_flexible( wgtDct.values( ) )

    # Polish
    bq.columns = [c.replace(' ', '<br>') if isinstance(c, str) else c for c in bq.iloc[0]]
    bq = bq.drop(df.index[0])
    bq = bq.reset_index(drop=True)
    model_names = bq.iloc[1:nRow+1, 0].tolist()
    mxCol0Char = max( ( max( [ len( m ) for m in model_names ] ) + 2 ) // 3 + 1 , 18 )
    bq.iloc[1:nRow+1, 0] = wrapLongString(model_names, max_len = mxCol0Char)
    # print( bq.columns , bq.loc[ 0 ] )  ###  

    return bq
'''
bq = getDfBoswellQuotientTable( dfn , df )

# float_cols = bq.select_dtypes(include='float').columns
# print( float_cols )
styled = bq.style.set_properties(**{'text-align': 'left', 'white-space': 'pre'})  #  .format("{:.1f}",subset=float_cols)
# dfi.export_png( styled , dir + 'BoswellQuotientTableOutput.png' )  #  for pd.__version__ > 2.1 and dfi.__version__ >= 0.2.7

export_and_display_table( styled , bq , saveTableImages , output_dir = dir , output_filename = "BoswellQuotientTableOutput.png" )
'''
#   =====

import argparse

def main( ) :
    parser = argparse.ArgumentParser( description = "Tabulate Boswell Quotients in Letter Grade for given grades_table.csv and timing_report.md." )
    parser.add_argument( '-d' , '--dir' , default = 'botwell/results/', help = "path to grades_table.csv and timing_report.md files" )
    parser.add_argument( '-w' , '--weight' , default = '5:2:2:1', help = "Boswell Quotient weights to be applied to each chatbot's normalized grade, accuracy, consistency, and latency; the ratios are normalied to weights" )

    args = parser.parse_args( )
    print( args )	#  args.dir , args.weight
    print( )

    dir = args.dir  
    if ( dir == 'botwell/results/' ) :
        dir = './results/20250228-ai_policy/'

    if ( dir[ -1 ] != '/' and dir[ -1 ].isalnum( ) ) :
        dir += '/'

    print( "python version : {sys.version.split( )[ 0 ]}" )
    print( "pandas version : {pd.__version__}" )
    print( "dataframe_image version : {dfi.__version__)" )
    print( "dir:\t\t" , dir )

    if ( not os.path.exists( dir ) ) :  #  isdir
        parser.error( f"Input file '{args.dir}' not found!" )
        print( "please supply directory path where 'grade_table.csv' and" )
        print( "'timing_report.md' resides; absolute path preferable.")

    wgts = args.weight
    npw = np.array( wgts.strip( ).split( ':' ) , dtype = float )
    npw /= np.sum( npw )
    dctWgt = dict( zip( [ 'normalized grades' , 'accuracy' , 'consistency' , 'latency' ] , npw.tolist( ) ) )
    wgt = wgts.replace( ':' , "" )
    print( f'weights : {wgts} {wgt}' )
    print( f'wgtDct : {dctWgt}' ) 
    print( )

    df0 = get_dataframe_with_latency( dir )

    nRow , nCol = df0.shape

#    print( "df :\n" , df0.columns , df0.shape , nRow , len( df0 ) , df0[ df0.columns[ : 2 ] ] , \
#        df0.iloc[ : , - 2 : ] )  #  ..., first and last 2 columns

    dfr = df0.copy( )
    model_names = dfr.iloc[:nRow, 0].tolist()
    mxCol0Char = max( ( max( [ len( m ) for m in model_names ] ) + 2 ) // 3 + 1 , 18 )
    dfr.iloc[:nRow, 0] = wrapLongString(model_names, max_len = mxCol0Char, wrapStr = '<br>' )
    # print( dfr.iloc[ : nRow , 0 ].tolist( ) )

    # float_cols = dfr.select_dtypes(include='float').columns
    # print( float_cols , dfr['latency'] )
    dfr['latency'] = dfr['latency'].apply(lambda x: int(round(x)) if not isinstance( x , str ) else x )
    
    styler = dfr.style.set_properties(**{'text-align': 'left', 'white-space': 'pre'})  #  .format("{:d}",subset=float_cols)

    export_and_display_table( styler, dfr, saveTableImages , output_dir = dir , output_filename = "RawGradeTableOutput" + wgt + ".png" )

#	=====

#    print( "df0 :\n" , df0.shape )

    dfn = getDfNormalizedGrades( df0 )

    dfnrm = dfn.copy( )
    model_names = dfnrm.iloc[:nRow, 0].tolist()
    mxCol0Char = max( ( max( [ len( m ) for m in model_names ] ) + 2 ) // 3 + 1 , 18 )
    dfnrm.iloc[:nRow, 0] = wrapLongString(model_names, max_len = mxCol0Char , wrapStr = '<br>' )
    dfnrm['latency'] = dfnrm['latency'].apply(lambda x: int(round(x)) if not isinstance( x , str ) else x ) 
    
    stylen = dfnrm.style.set_properties(**{'text-align': 'left', 'white-space': 'pre'})  #  .format("{:.1f}",subset=float_cols)

    export_and_display_table( stylen , dfnrm , saveTableImages , output_dir = dir , output_filename = "NormalizedGradeTableOutput" + wgt + ".png" )

#	=====

    bq = getDfBoswellQuotientTable( dfn , df0 , wgtDct = dctWgt )

    styled = bq.style.set_properties(**{'text-align': 'left', 'white-space': 'pre'})  #  .format("{:.1f}",subset=float_cols)
    # dfi.export_png( styled , dir + 'BoswellQuotientTableOutput.png' )  #  for pd.__version__ > 2.1 and dfi.__version__ >= 0.2.7

    export_and_display_table( styled , bq , saveTableImages , output_dir = dir , output_filename = "BoswellQuotientTableOutput" + wgt + ".png" )

#	=====

if ( __name__ == "__main__" ) :
    main( )
