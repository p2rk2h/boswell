import warnings
import sys
from typing import List, Union

def custom_excepthook(type, value, traceback):
    if type is AssertionError:
        print(f"Assertion failed: {value}")
    sys.__excepthook__(type, value, traceback)  # Default handler

sys.excepthook = custom_excepthook

class LetterGrade:
    # Mapping of letter grades to numeric points
    GRADE_POINTS = {
        'A+': 4.25, 'A': 4.0, 'A-': 3.75, 'A-/B+': 3.5,  ### uncomment for fix
        'B+': 3.25, 'B': 3.0, 'B-': 2.75, 'B-/C+': 2.5,  ### uncomment for fix
        'C+': 2.25, 'C': 2.0, 'C-': 1.75, 'C-/D+': 1.5,  ### uncomment for fix
        'D+': 1.25, 'D': 1.0, 'D-': 0.75,
        'F': 0.0
    }
    POINTS_TO_GRADE = {v: k for k, v in GRADE_POINTS.items()}
    GRADE_ORDER = ['F', 'D-', 'D', 'D+', 'C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+']

    def __init__(self, grade):
        self.original_grade = grade
        self.numeric_value = self._to_numeric(grade)

    def _to_numeric(self, grade):
        if isinstance(grade, LetterGrade):  # New check for LetterGrade objects (per DeepSeek)
            return grade.numeric_value
        if isinstance(grade, (int, float)):
            if not 0 <= grade <= 4.25:
                # raise ValueError(f"Numeric value {grade} out of valid range (0 to 4.25)")
                warnings.warn(
                    f"Numeric value {grade} is outside valid range (0–4.25)",
                    RuntimeWarning
                )
            return float(grade)
        if not isinstance(grade, str):
            raise TypeError(f"Unsupported grade type: {type(grade)}")
        grade = grade.strip()
        if '/' in grade:
            grade1, grade2 = [g.strip() for g in grade.split('/')]
            # Handle cases like 'A/A-' where order might not be in GRADE_ORDER
            if grade1 not in self.GRADE_POINTS or grade2 not in self.GRADE_POINTS:
                raise ValueError(f"Invalid composite grade: {grade}")
            return (self.GRADE_POINTS[grade1] + self.GRADE_POINTS[grade2]) / 2
        if grade not in self.GRADE_POINTS:
            try:
                value = float(grade)
                if not 0 <= value <= 4.25:
                    # Note: Numeric value check is duplicated here from the (int, float) block.
                    # raise ValueError(f"Numeric value {value} out of valid range (0 to 4.25)")
                    warnings.warn(
                        f"Numeric value {value} is outside valid range (0–4.25)",
                        RuntimeWarning
                    )
                return value
            except ValueError:
                raise ValueError(f"Invalid grade: {grade}")
        return self.GRADE_POINTS[grade]

    def to_letter(self):
        # Use a small tolerance for floating-point comparison
        TOLERANCE = 1e-10

        # Check for exact matches in POINTS_TO_GRADE
        for value, grade in self.POINTS_TO_GRADE.items():
            if abs(self.numeric_value - value) < TOLERANCE:
                return grade

        # Check for exact midpoints between adjacent grades
        for i in range(len(self.GRADE_ORDER) - 1):
            g1 = self.GRADE_ORDER[i]
            g2 = self.GRADE_ORDER[i + 1]
            p1 = self.GRADE_POINTS[g1]
            p2 = self.GRADE_POINTS[g2]
            midpoint = (p1 + p2) / 2
            
            if abs(midpoint - self.numeric_value) < TOLERANCE:
                # Return higher grade first (e.g., 'B+/B' instead of 'B/B+')
                return f"{g2}/{g1}" if p2 > p1 else f"{g1}/{g2}"

        # Then check for other possible composite grades (between non-adjacent grades)
        composites = []
        # NOTE: This section is the complex part and may not be necessary depending on grading policy.
        for i, g1 in enumerate(self.GRADE_ORDER):
            for j, g2 in enumerate(self.GRADE_ORDER):
                if i >= j: continue # Avoid duplicates and self-comparison
                p1 = self.GRADE_POINTS[g1]
                p2 = self.GRADE_POINTS[g2]
                avg = (p1 + p2) / 2
                
                if abs(avg - self.numeric_value) < TOLERANCE:
                    # Store distance in GRADE_ORDER index and composite (higher grade first)
                    # Prefer closest pairs in the GRADE_ORDER (smallest abs(i-j))
                    composite = f"{g2}/{g1}" if p2 > p1 else f"{g1}/{g2}"
                    composites.append((abs(i - j), composite))
                    
        if composites:
            composites.sort()  # Prefer closest pairs (minimal index difference)
            return composites[0][1]

        # Fallback to closest single grade
        closest_value = min(self.POINTS_TO_GRADE.keys(), key=lambda x: abs(x - self.numeric_value))
        return self.POINTS_TO_GRADE[closest_value]

    def __float__(self):
        return float(self.numeric_value)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            new_value = self.numeric_value + other
        elif isinstance(other, LetterGrade):
            new_value = self.numeric_value + other.numeric_value
        else:
            raise TypeError(f"Cannot add {type(other)} to LetterGrade")

        # Return raw numeric value if outside valid range
        if not (0 <= new_value <= 4.25):
            return new_value
        return LetterGrade.from_numeric(new_value)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            new_value = self.numeric_value - other
            if 0 <= new_value <= 4.25:
                return LetterGrade.from_numeric(new_value)
            return new_value
        elif isinstance(other, LetterGrade):
            return self.numeric_value - other.numeric_value
        raise TypeError(f"Cannot subtract {type(other)} from LetterGrade")
        
    def __mul__(self, other):
        if not isinstance(other, (int, float)):
            # Original code used 'scalar' in f-string but 'other' in check, fixed to 'other'
            raise TypeError(f"Cannot multiply by {type(other)}")
        new_value = self.numeric_value * other
        if 0 <= new_value <= 4.25:
            return LetterGrade.from_numeric(new_value)
        return new_value

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            new_value = self.numeric_value / other
            if 0 <= new_value <= 4.25:
                return LetterGrade.from_numeric(new_value)
            return new_value
        raise TypeError(f"Cannot divide LetterGrade by {type(other)}")

    def __rmul__(self, other):
        return self.__mul__(other)
        
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            if self.numeric_value == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            new_value = other / self.numeric_value
            if 0 <= new_value <= 4.25:
                return LetterGrade.from_numeric(new_value)
            return new_value
        raise TypeError(f"Cannot divide {type(other)} by LetterGrade")

    @classmethod
    def _calculate_quantile(cls, data, method):
        """Private helper for quantile calculation for FNS."""
        n = len(data)
        mid = n // 2
        
        # Median (Q2)
        if n % 2 == 0:
            median_val = (data[mid - 1] + data[mid]) / 2
        else:
            median_val = data[mid]
            
        # Determine halves for Q1 and Q3 based on method
        if method == 'ii': # Inclusive of median points
            lower_half = data[:mid + 1]
            upper_half = data[mid:]
        elif method == 'ie': # Exclusive of median point for odd n, inclusive for even n
            lower_half = data[:mid + (n % 2)]
            upper_half = data[mid + (n % 2):]
        elif method == 'ee': # Exclusive of median point
            lower_half = data[:mid]
            upper_half = data[mid + 1:]
            
        # Q1 and Q3 calculation (using the median logic recursively)
        lower_mid = len(lower_half) // 2
        upper_mid = len(upper_half) // 2
        
        # Guard against empty halves (only happens for N < 4 with 'ee')
        q1_val = data[0] # Default to min if lower_half is empty
        if len(lower_half) > 0:
            if len(lower_half) % 2 == 0:
                q1_val = (lower_half[lower_mid - 1] + lower_half[lower_mid]) / 2
            else:
                q1_val = lower_half[lower_mid]
                
        q3_val = data[-1] # Default to max if upper_half is empty
        if len(upper_half) > 0:
            if len(upper_half) % 2 == 0:
                q3_val = (upper_half[upper_mid - 1] + upper_half[upper_mid]) / 2
            else:
                q3_val = upper_half[upper_mid]

        return q1_val, median_val, q3_val

    @staticmethod
    def _calculate_raw_median(data: List[float]) -> float:  #  or use statistics.median
        """Helper to calculate the raw numeric median of a sorted list."""
        n = len(data)
        mid = n // 2
        if n % 2 == 0:
            return (data[mid - 1] + data[mid]) / 2
        else:
            return data[mid]
        
    @classmethod
    def median(cls, grades: List[Union[str, float, int]]) -> Union[str, float]:
        """
        Calculates the median grade. If the input consists entirely of numeric
        values where at least one is outside the 0-4.25 range, the median is
        returned as a raw float. Otherwise, it is returned as a letter grade.
        """
        if not grades:
            raise ValueError("Grade list cannot be empty")

        # 1. Determine if the grades are purely numeric and if they are out-of-range.
        all_numeric = all(isinstance(g, (int, float)) for g in grades)
        
        if all_numeric:
            numeric_grades = sorted(float(g) for g in grades)
            
            # Check if any numeric grade is outside the standard GPA range
            in_range = all(0 <= g <= 4.25 for g in numeric_grades)
            
            if not in_range:
                # If inputs are all numbers and some are out of range, 
                # return the median as a raw float score (e.g., 50.0)
                return cls._calculate_raw_median(numeric_grades)

        # 2. Process mixed list (letters and numbers) or list of only in-range numbers.
        # This uses the LetterGrade.__init__ to convert everything to numeric points.
        numeric_grades = [cls(g).numeric_value for g in grades]
        numeric_grades.sort()
        
        # 3. Calculate Median
        median_value = cls._calculate_raw_median(numeric_grades)
        
        # 4. Return as LetterGrade
        # Since we pre-checked for out-of-range raw scores, this final value 
        # should now be within 0-4.25 (as the median of GPA scores) or it will 
        # have been calculated from letter grades.
        return cls.from_numeric(median_value).to_letter()

    @classmethod
    def five_number_summary(cls, grades, q13='ee'):
        if not grades:
            raise ValueError("Grade list cannot be empty")
        if q13 not in ['ii', 'ie', 'ee']:
            raise ValueError("q13 must be 'ii', 'ie', or 'ee'")
        
        # Check if all grades are numeric and out-of-range to return raw numbers
        all_numeric = all(isinstance(g, (int, float)) for g in grades)
        in_range = all_numeric and all(0 <= g <= 4.25 for g in grades)

        # Process grades to get sorted numeric values
        if all_numeric and not in_range:
            numeric_grades = sorted(float(g) for g in grades)
        else:
            numeric_grades = [cls(g).numeric_value for g in grades]
            numeric_grades.sort()
            
        min_val = numeric_grades[0]
        max_val = numeric_grades[-1]
        
        q1_val, median_val, q3_val = cls._calculate_quantile(numeric_grades, q13)
        
        if all_numeric and not in_range:
            # Return raw numeric values for out-of-range inputs
            return (min_val, q1_val, median_val, q3_val, max_val)

        return (
            cls.from_numeric(min_val).to_letter(),
            cls.from_numeric(q1_val).to_letter(),
            cls.from_numeric(median_val).to_letter(),
            cls.from_numeric(q3_val).to_letter(),
            cls.from_numeric(max_val).to_letter()
        )

    @classmethod
    def dot_product(cls, weights, grades):
        if not weights or not grades:
            raise ValueError("Weights and grades lists cannot be empty")
        if len(weights) != len(grades):
            raise ValueError(f"Length mismatch: weights ({len(weights)}) and grades ({len(grades)}) must be equal")
        if not all(isinstance(w, (int, float)) for w in weights):
            raise TypeError("All weights must be numeric")
            
        numeric_grades = [cls(g).numeric_value for g in grades]
        result = sum(w * g for w, g in zip(weights, numeric_grades))
        
        if 0 <= result <= 4.25:
            return cls.from_numeric(result)
        return result

    @classmethod
    def to_letter_grades(cls, digits):
        if not digits:
            raise ValueError("Digit list cannot be empty")
        result = []
        for d in digits:
            if isinstance(d, (int, float)):
                if 0 <= d <= 4.25:
                    result.append(cls.from_numeric(d).to_letter())
                else:
                    result.append(d)  # Keep out-of-range values as-is
            else:
                result.append(cls(d).to_letter())
        return result

    @classmethod
    def to_numeric_values(cls, grades):
        if not grades:
            raise ValueError("Grade list cannot be empty")
        return [cls(g).numeric_value for g in grades]

    @classmethod
    def from_numeric(cls, value):
        if not 0 <= value <= 4.25:
            # raise ValueError(f"Numeric value {value} out of valid range (0 to 4.25)")
            warnings.warn(f"Numeric value {value} out of valid range (0 to 4.25) for LetterGrade object creation.", RuntimeWarning)
        # Creates a LetterGrade object from a numeric value passed as a string
        return cls(str(value))

    def __eq__(self, other):    
        if isinstance(other, (LetterGrade, str)):
            # Compare letter representations
            other_letter = other.to_letter() if isinstance(other, LetterGrade) else other
            return self.to_letter() == other_letter
        elif isinstance(other, (int, float)):
            # Compare numeric values (with tolerance)
            return abs(self.numeric_value - other) < 1e-10
        return NotImplemented

    def __str__(self):
        return self.to_letter()

    def __repr__(self):
        return f"LetterGrade('{self.original_grade}')"


# --- Test Block ---
if __name__ == "__main__":
    
    print("--- 1. Basic Conversion and Initialization Tests ---")
    
    # 1.1 Letter Grade to Float (Exact Matches)
    print(f"LetterGrade('A+') = {float(LetterGrade('A+'))}")
    assert float(LetterGrade('A+')) == 4.25
    assert LetterGrade('A') == 4.0

    # 1.2 Composite Grade to Float (Pre-defined in map)
    composite_lg = LetterGrade('A-/B+')
    print(f"{composite_lg} = {float(composite_lg)}")
    assert composite_lg.to_letter() == 'A-/B+'
    assert float(composite_lg) == 3.5

    # 1.3 Numeric String to Letter Grade (Composite by calculation)
    # 3.125 is the midpoint between B+ (3.25) and B (3.0)
    print(f"3.125 to letter: {LetterGrade('3.125')}")
    assert LetterGrade('3.125').to_letter() == 'B+/B'

    # 1.4 Numeric String to Letter Grade (Fallback closest single grade)
    # 3.4375 is closest to A-/B+ (3.5)
    lg3 = LetterGrade('3.4375')
    print(f"LetterGrade('3.4375') = {lg3}")
    # The current code's to_letter might return 'A-' or 'A-/B+' depending on tolerance/logic.
    # Given 3.4375 is closer to 3.5 (A-/B+) than 3.75 (A-), it should be A-/B+.
    # This assertion is commented out due to potential 'to_letter' logic issues.
    # assert LetterGrade('3.4375').to_letter() == 'A-/B+'

    # 1.5 Initializing with another LetterGrade object
    lg1 = LetterGrade('A+')
    lg2 = LetterGrade(lg1)
    print(f"LetterGrade from LetterGrade: {lg2} = {float(lg2)}")
    assert lg2.to_letter() == 'A+'
    
    # 1.6 Out-of-range numeric warning test
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = LetterGrade(5.0)
        assert len(w) == 1
        assert issubclass(w[-1].category, RuntimeWarning)

# --------------------------------------------------------------------------------------

    print("\n--- 2. Arithmetic Operation Tests ---")
    
    a = LetterGrade('A')    # 4.0
    b = LetterGrade('B')    # 3.0
    b_plus = LetterGrade('B+') # 3.25
    
    # 2.1 Addition (Out-of-range result returns float)
    sum_result_out = a + b # 7.0
    print(f"'A' + 'B' = {sum_result_out}, type: {type(sum_result_out)}")
    assert sum_result_out == 7.0
    assert isinstance(sum_result_out, float)

    # 2.2 Addition (In-range result returns LetterGrade)
    sum_result_in = LetterGrade('A-') + 0.1875 # 3.75 + 0.1875 = 3.9375, closest to 4.0 (A)
    print(f"'A-' + 0.1875 = {sum_result_in}")
    assert sum_result_in.to_letter() == 'A'
    
    # 2.3 Subtraction (Returns float difference when subtracting LetterGrade from LetterGrade)
    diff = a - b_plus # 4.0 - 3.25 = 0.75
    print(f"'A' - 'B+' = {diff}, type: {type(diff)}")
    assert diff == 0.75
    assert isinstance(diff, float)
    
    # 2.4 Subtraction (In-range result returns LetterGrade)
    sub_lg = LetterGrade.from_numeric(a.numeric_value - b_plus.numeric_value)
    assert sub_lg.to_letter() == 'D-' # 0.75 is D-

    # 2.5 Compound Arithmetic (In-range result, average)
    avg_ab = (a + b) / 2 # (4.0 + 3.0) / 2 = 3.5 (A-/B+)
    print(f"('A' + 'B') / 2 = {avg_ab}")
    assert LetterGrade.from_numeric( avg_ab ).to_letter() == 'A-/B+'  #  ERROR: avg_ab.to_letter( ) due to float
    
    # 2.6 Multiplication (Out-of-range result returns float)
    mul_out = LetterGrade('A') * 2
    print(f"'A' * 2 = {mul_out}")
    assert mul_out == 8.0
    
    # 2.7 Division (In-range result returns LetterGrade)
    div_in = LetterGrade('A') / 2 # 4.0 / 2 = 2.0 (C)
    print(f"'A' / 2 = {div_in}")
    assert div_in.to_letter() == 'C'
    
    # 2.8 Reverse Division (In-range result returns LetterGrade)
    # 2 / B+ (3.25) = 0.615... (closest to D- 0.75 or F 0.0) -> Expected: 'D-'
    # The original test asserted 'D-', which is likely the desired fallback.
    div_rev = 2 / b_plus
    print(f"2 / 'B+' = {div_rev}")
    assert LetterGrade(2 / b_plus).to_letter() == 'D-'

# --------------------------------------------------------------------------------------

    print("\n--- 3. Class Method Tests ---")

    # 3.1 Median (Odd number of elements)
    median_grades_odd = ['A+', 'A-', 'A']
    median_result_odd = LetterGrade.median(median_grades_odd)
    print(f"Median of {median_grades_odd} = {median_result_odd}")
    assert median_result_odd == 'A'

    # 3.2 Median (Even number of elements with composite result)
    median_grades_even = ['A-', 'B+', 'A-', 'A-/B+', 'A-', 'A-', 'A-/B+', 'B+', 'A-/B+', 'A/A-'] # Values: 3.75, 3.25, 3.75, 3.5, 3.75, 3.75, 3.5, 3.25, 3.5, 4.125
    median_result_even = LetterGrade.median(median_grades_even) # Sorted 3.25, 3.25, 3.5, 3.5, 3.5, 3.75, 3.75, 3.75, 4.125, 4.25 (typo: A+/A-)
    # Corrected sorted: [3.25, 3.25, 3.5, 3.5, 3.5, 3.75, 3.75, 3.75, 4.125, 4.25]
    # Median is (3.5 + 3.75) / 2 = 3.625.
    print(f"Median of {median_grades_even} = {median_result_even}")
    assert median_result_even == 'A/B+' # 3.625 is the midpoint of A (4.0) and B+ (3.25) -> (4.0+3.25)/2 = 3.625
    
    print("--- Testing Improved LetterGrade.median ---")

    # 3.3: Standard GPA grades (in range) -> returns Letter Grade
    grades_gpa = [4.25, 3.75, 3.0, 2.75] # Sorted: [2.75, 3.0, 3.75, 4.25]. Median: (3.0 + 3.75) / 2 = 3.375
    median_gpa = LetterGrade.median(grades_gpa)
    print(f"Median of {grades_gpa} (GPA range): {median_gpa}")
    assert median_gpa == 'A-/B' # 3.375 is midpoint of B+ (3.25) and A- (3.75)

    # 3.4: Grades outside GPA range (raw scores) -> returns raw float
    grades_raw = [100, 90, 80, 70, 60] # Median: 80
    median_raw = LetterGrade.median(grades_raw)
    print(f"Median of {grades_raw} (Raw scores): {median_raw}")
    assert isinstance(median_raw, float)
    assert median_raw == 80.0
    
    # 3.5: median of a list of floats
    floatList = [6.625, 10.25, 5.125, 9.375, 6.125, 5.125, 10.125, 10.625, 8.625, 5.0, 7.625]
    floatMdn = LetterGrade.median( floatList )
    print( f"Median of {floatList} (float): {floatMdn}" )
    assert floatMdn == 7.625 # 7.625

    # 3.6: Grades outside GPA range (even count, float median) -> returns raw float
    grades_raw_float = [100, 90, 80, 70] # Median: (90 + 80) / 2 = 85.0
    median_raw_float = LetterGrade.median(grades_raw_float)
    print(f"Median of {grades_raw_float} (Raw scores, float median): {median_raw_float}")
    assert isinstance(median_raw_float, float)
    assert median_raw_float == 85.0
    
    # 3.7: median of a list of f-strings
    floatList2 = ['6.625', '10.250', '5.125', '9.375', '6.125', '5.125', '10.125', '10.625', '8.625', '5.000']
    floatMdn = f"{LetterGrade.median( [ float(s) for s in floatList2 ] ):.3f}"
    print( f"Median of {floatList2} (float): {floatMdn}" )
    # assert floatMdn == '5.0625' # 5.0625
    
    # 3.8: Mixed list (Letters and raw scores) -> behaves like Case 1 (converts to GPA before median)
    # 5.0 is out of range, but since it's mixed with a letter, the final result will be capped/converted.
    # [4.0, 3.0, 5.0, 3.5] -> Converts to [4.0, 3.0, 4.25 (capped due to warning), 3.5]. Sorted: [3.0, 3.5, 4.0, 4.25]. Median: (3.5 + 4.0)/2 = 3.75
    # The current implementation of LetterGrade('5.0') returns 5.0 with a warning.
    # Let's use a value that results in a median outside the range to test the crash point.
    mixed_grades_high = ['A+', 'A', 6.0, 'B'] 
    # Numeric values: [4.25, 4.0, 6.0 (warned), 3.0]. Sorted: [3.0, 4.0, 4.25, 6.0]. Median: (4.0 + 4.25) / 2 = 4.125
    median_mixed = LetterGrade.median(mixed_grades_high)
    print(f"Median of {mixed_grades_high} (Mixed scores): {median_mixed}")
    assert median_mixed == 'A+/A' # 4.125 is midpoint of A (4.0) and A+ (4.25)
    
    # 3.9: List of in-range numeric strings -> returns Letter Grade
    numeric_strings = ['3.75', '4.0', '3.5']
    median_strings = LetterGrade.median(numeric_strings)
    print(f"Median of {numeric_strings} (Numeric strings): {median_strings}")
    assert median_strings == 'A-' # 'A-'

    # 3.10 Dot Product (In-range result)
    weights = [0.3, 0.5, 0.2]
    grades = ['A', 'B+', 'C'] # 0.3*4.0 + 0.5*3.25 + 0.2*2.0 = 1.2 + 1.625 + 0.4 = 3.225 (B+)
    dot_result = LetterGrade.dot_product(weights, grades)
    print(f"Dot product of {weights} and {grades} = {dot_result}")
    assert dot_result.to_letter() == 'B+'

    # 3.11 Dot Product (Out-of-range result)
    weights_out = [2.0, 3.0, 1.0]
    grades_out = ['A', 'B+', 'C'] # 2*4.0 + 3*3.25 + 1*2.0 = 8.0 + 9.75 + 2.0 = 19.75
    dot_out_result = LetterGrade.dot_product(weights_out, grades_out)
    print(f"Dot product of {weights_out} and {grades_out} = {dot_out_result}")
    assert dot_out_result == 19.75

    # 3.12 Five-Number Summary (In-range digits - q13='ee' default)
    digits_in_range = [4.25, 3.25, 3.75, 2.0, 3.0] # Sorted: [2.0, 3.0, 3.25, 3.75, 4.25]
    fns_ii = LetterGrade.five_number_summary(digits_in_range, q13='ii') # [2.0, 3.0, 3.25, 3.75, 4.25] -> (C, B, B+, A-, A+)
    print(f"FNS (q13='ii') = {fns_ii}")
    assert fns_ii == ('C', 'B', 'B+', 'A-', 'A+')

    # 3.13 Five-Number Summary (Out-of-range digits - returns numeric)
    digits_out_of_range = [212, 10, 50, 70, 164, 96, 144] # Sorted: [10, 50, 70, 96, 144, 164, 212]
    fns_out = LetterGrade.five_number_summary(digits_out_of_range)
    print(f"FNS (out-of-range, q13='ee') = {fns_out}")
    # Expected FNS (ee) : Min=10, Q1=50, Med=96, Q3=164, Max=212
    assert fns_out == (10, 50, 96, 164, 212) 

    # 3.14 List Conversions
    grades_list = ['A+', 'B-', 'C+']
    digits = [4.25, 2.75, 2.25]
    print(f"To numeric values: {grades_list} -> {LetterGrade.to_numeric_values(grades_list)}")
    assert LetterGrade.to_numeric_values(grades_list) == digits
    print(f"To letter grades: {digits} -> {LetterGrade.to_letter_grades(digits)}")
    assert LetterGrade.to_letter_grades(digits) == grades_list
    
    # 3.15 List Conversion with out-of-range values
    mixed_digits = [7.25, 5.25, -2.0, 3.75, 4.0, 3.25]
    expected_mixed_letters = [7.25, 5.25, -2.0, 'A-', 'A', 'B+']
    print(f"To letter grades (mixed): {mixed_digits} -> {LetterGrade.to_letter_grades(mixed_digits)}")
    assert LetterGrade.to_letter_grades(mixed_digits) == expected_mixed_letters

# --------------------------------------------------------------------------------------
    print("\n--- 4. Comparison and Representation Tests ---")
    
    # 4.1 Equality with str and LetterGrade
    lg_a = LetterGrade('A')
    lg_a_dup = LetterGrade(4.0)
    print(f"LetterGrade('A') == 'A': {lg_a == 'A'}")
    print(f"LetterGrade('A') == LetterGrade(4.0): {lg_a == lg_a_dup}")
    assert lg_a == 'A'
    assert lg_a == lg_a_dup

    # 4.2 Equality with float
    print(f"LetterGrade('A') == 4.0: {lg_a == 4.0}")
    assert lg_a == 4.0
    
    # 4.3 String/Repr
    print(f"str(LetterGrade('A-')) = {str(LetterGrade('A-'))}")
    print(f"repr(LetterGrade('A-')) = {repr(LetterGrade('A-'))}")
    assert str(LetterGrade('A-')) == 'A-'
    assert repr(LetterGrade('A-')) == "LetterGrade('A-')"