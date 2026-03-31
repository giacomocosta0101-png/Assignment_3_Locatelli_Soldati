"""
Mathematical Engineering - Financial Engineering, FY 2024-2025
Risk Management - Exercise 2: Corporate Bond Portfolio
"""

from typing import List, Union
import numpy as np
import pandas as pd
import datetime as dt
from utilities.ex1_utilities import (
  
    get_discount_factor_by_zero_rates_linear_interp,
)
from utilities.date_functions import (
    business_date_offset,
    year_frac_act_x,
    year_frac_30e_360,
    schedule_year_fraction
)


def bond_payment_dates(
    issue_date: Union[dt.date, pd.Timestamp], maturity: int, coupon_freq: int
) -> Union[List[dt.date], List[pd.Timestamp]]:
    """
    Calculate the payment dates of a bond.

    Parameters:
        issue_date (Union[dt.date, pd.Timestamp]): Bond's issue date.
        maturity (int): Bond's maturity in years.
        coupon_freq (int): Coupon frequency in payments per years.

    Returns:
        Union[List[dt.date], List[pd.Timestamp]]: List of payment dates.
    """

    payment_dates = []
    counter = 1
    for _ in range(maturity * coupon_freq):
        payment_dt = business_date_offset(
            issue_date, month_offset=(12 // coupon_freq) * counter
        )
        
        payment_dates.append(payment_dt)

        counter += 1

    return payment_dates




def bond_cash_flows(
    ref_date: Union[dt.date, pd.Timestamp],
    issue_date: Union[dt.date, pd.Timestamp],
    maturity: int,
    coupon_rate: float,
    coupon_freq: int,
    notional: float = 1.0,
) -> pd.Series:
    """
    Calculate the cash flows of a bond.

    Parameters:
    ref_date (Union[dt.date, pd.Timestamp]): Reference date.
    issue_date (Union[dt.date, pd.Timestamp]): Bond's issue date.
    maturity (int): Bond's maturity in years.
    coupon_rate (float): Coupon rate.
    coupon_freq (int): Coupon frequency in payments per years.
    notional (float): Notional amount.

    Returns:
        pd.Series: Bond cash flows.
    """

    # Payment dates
    cash_flows_dates = bond_payment_dates(issue_date, maturity, coupon_freq)
   


    # Coupon payments
    dates = [ref_date] + cash_flows_dates
    
    yf_between_dates = schedule_year_fraction(dates)
    cash_flows = pd.Series(
        data=[coupon_rate * notional * yf for yf in yf_between_dates],
        index=cash_flows_dates)


    # Notional payment
    cash_flows[cash_flows_dates[-1]] += notional

    return cash_flows


def defaultable_bond_dirty_price_from_intensity(
    ref_date: Union[dt.date, pd.Timestamp],
    issue_date: Union[dt.date, pd.Timestamp],
    maturity: int,
    coupon_rate: float,
    coupon_freq: int,
    recovery_rate: float,
    intensity: Union[float, pd.Series],
    discount_factors: pd.Series,
    notional: float = 1.0,
) -> float:
    """
    Calculate the dirty price of a defaultable bond neglecting the recovery of the coupon payments.

    Parameters:
    ref_date (Union[dt.date, pd.Timestamp]): Reference date.
    issue_date (Union[dt.date, pd.Timestamp]): Bond's issue date.
    maturity (int): Bond's maturity in years.
    coupon_rate (float): Coupon rate.
    coupon_freq (int): Coupon frequency in payments a years.
    recovery_rate (float): Recovery rate.
    intensity (Union[float, pd.Series]): Intensity, can be the average intensity (float) or a
        piecewise constant function of time (pd.Series).
    discount_factors (pd.Series): Discount factors.
    notional (float): Notional amount.

    Returns:
        float: Dirty price of the bond.
    """

    # Calculate the cash flows
    cash_flows = bond_cash_flows(
        ref_date, issue_date, maturity, coupon_rate, coupon_freq, notional
    )
  

    # Discount factors
    discount_factors = [get_discount_factor_by_zero_rates_linear_interp(ref_date,cp_date,
                                                                       discount_factors.index,discount_factors.values) 
                                                                       for cp_date in cash_flows.index]

    # Calculate the survival probabilities and default probabilities
    if isinstance(intensity, float):                 # if intensity is a float -> constant value
        survival_probs = np.exp(
            [
                -intensity * year_frac_act_x(ref_date, date, 365)
                for date in cash_flows.index
            ]
        )
        survival_probs = pd.Series(data=survival_probs, index=cash_flows.index)
    else:                                            # if instensity is not a float -> pd.Series -> piecewise constant
    
        survival_probs_list = []

        for date in cash_flows.index:
            integral = 0.0
            prev = ref_date
            for k in range(len(intensity)):
                t_k = intensity.index[k]
                lam_k = intensity.values[k]
                
                # estremo superiore del segmento: il minore tra t_k e la data del cash flow
                segment_end = min(date, t_k)
                yf = year_frac_act_x(prev, segment_end, 365)
                integral += lam_k * yf
                
                prev = t_k
                
                # se la data cade dentro (o al bordo di) questo segmento, ho finito
                if date <= t_k:
                    break
                    
            survival_probs_list.append(np.exp(-integral))
        survival_probs = pd.Series(data=survival_probs_list, index=cash_flows.index)

    all_surv = [1.0] + list(survival_probs.values)
    
    default_probs = pd.Series(
        data=[all_surv[i] - all_surv[i + 1] for i in range(len(survival_probs))],
        index=cash_flows.index
    )

    # Calculate the dirty price
    # Coupon + principal leg (weighted by survival)
    dirty_price = sum(cf * df * sp 
                    for cf, df, sp in zip(cash_flows.values, discount_factors, survival_probs.values))
    
    # Recovery leg (weighted by default probability)
    dirty_price += recovery_rate * notional * sum(df * dp 
                    for df, dp in zip(discount_factors, default_probs.values))
    
    return dirty_price


def defaultable_bond_dirty_price_from_z_spread(
    ref_date: Union[dt.date, pd.Timestamp],
    issue_date: Union[dt.date, pd.Timestamp],
    maturity: int,
    coupon_rate: float,
    coupon_freq: int,
    z_spread: float,
    discount_factors: pd.Series,
    notional: float = 1.0,
) -> float:
    """
    Calculate the dirty price of a defaultable bond from the Z-spread.

    Parameters:
    ref_date (Union[dt.date, pd.Timestamp]): Reference date.
    issue_date (Union[dt.date, pd.Timestamp]): Bond's issue date.
    maturity (int): Bond's maturity in years.
    coupon_rate (float): Coupon rate.
    coupon_freq (int): Coupon frequency in payments a years.
    z_spread (float): Z-spread.
    discount_factors (pd.Series): Discount factors.
    notional (float): Notional amount.

    Returns:
        float: Dirty price of the bond.
    """

    # Calculate the cash flows
    cash_flows = bond_cash_flows(
        ref_date, issue_date, maturity, coupon_rate, coupon_freq, notional
    )
    
    # Discount factors
    discount_factors = [get_discount_factor_by_zero_rates_linear_interp(ref_date,cp_date,
                                                                       discount_factors.index,discount_factors.values) 
                                                                       for cp_date in cash_flows.index]

    # Calculate the survival probabilities and default probabilities
    survival_probs = np.exp(
            [
                -z_spread * year_frac_act_x(ref_date, date, 365)
                for date in cash_flows.index
            ]
        )
    survival_probs = pd.Series(data=survival_probs, index=cash_flows.index)


    # Discount factors with z-spread
    discount_factors = [df*sp for df,sp in zip(discount_factors,survival_probs.values)]

    # Calculate the dirty price
    dirty_price = sum(cf * df 
                    for cf, df in zip(cash_flows.values, discount_factors))
    
    return dirty_price