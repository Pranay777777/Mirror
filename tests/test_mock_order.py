import unittest
from unittest.mock import patch
import math

class TestMockOrder(unittest.TestCase):
    @patch('math.sin') # Top     -> Expected Last Arg
    @patch('math.cos') # Bottom  -> Expected First Arg
    def test_order(self, mock_cos, mock_sin):
        # By standard docs: Bottom is First arg. Top is Last arg.
        
        mock_cos.return_value = 999
        mock_sin.return_value = 888
        
        c = math.cos(0)
        s = math.sin(0)
        
        print(f"Cos returned: {c}")
        print(f"Sin returned: {s}")
        
        # If mock_cos corresponds to math.cos (bottom patch), c should be 999.
        self.assertEqual(c, 999, "Arg 1 should be Cos (Bottom)")
        self.assertEqual(s, 888, "Arg 2 should be Sin (Top)")
        
if __name__ == '__main__':
    unittest.main()
