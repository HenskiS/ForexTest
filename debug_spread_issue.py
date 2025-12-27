"""
Debug spread calculation difference
"""

# Example: USDJPY trade
exit_price_raw = 110.00
spread = 0.02  # 2 pips for JPY pair

# Working script approach (CORRECT)
exit_price_working = exit_price_raw - spread
print(f"Working script (subtract): {exit_price_raw} - {spread} = {exit_price_working}")
print(f"  Spread cost: {(exit_price_raw - exit_price_working) / exit_price_raw * 100:.4f}%")

# SL/TP script approach (INCORRECT)
exit_price_sltp = exit_price_raw * (1 - spread)
print(f"\nSL/TP script (multiply): {exit_price_raw} * (1 - {spread}) = {exit_price_sltp}")
print(f"  Spread cost: {(exit_price_raw - exit_price_sltp) / exit_price_raw * 100:.4f}%")

print(f"\nDifference: {exit_price_working - exit_price_sltp:.4f}")
print(f"Per trade impact on $10k position: ${10000 * (exit_price_working - exit_price_sltp) / exit_price_raw:.2f}")
