# Truth Finder
> Data fusion and truth discovery from multiple conflicting information providers

## Overview
This work implements the TruthFinder algorithm as presented in the paper [Truth Discovery with Multiple Conflicting Information Providers on the Web](http://web.cs.ucla.edu/~yzsun/classes/2014Spring_CS7280/Papers/Trust/kdd07_xyin.pdf). TruthFinder is an algorithm that tries to solve the veracity problem in an iterative process, thus find the most probable true value for an object given a set of possible conflicting values.

#### [Documentatation](docs/report.pdf)
#### [Slides](docs/slides.pdf)

# Prerequisites
* Python 3.7 or higher
* Virtualenv
* Microsoft Visual C++ 14.0 for `fuzzywuzzy[speedup]`, otherwise just use `fuzzywuzzy`

## Installation
```sh
$ git clone https://github.com/nhabbash/truth_discovery
$ cd truth_discovery
$ python -m venv .venv
$ .venv/Scripts/Activate.ps1
$ pip install -r requirements
```

## Usage
```python
import pandas as pd
from truthfinder import TruthFinder

# Read DataFrame, eventually clean it

f = "attribute"
obj = "identifier"
finder = TruthFinder(dataframe.copy(deep=True), fact=f, obj=obj)
result = finder.compute()

# Result ontains the maximum confidence values for fact/attribute f of unique object obj
print(result) 
```
Check the notebook for more on its use
#
## Authors
* **Nassim Habbash** - [nhabbash](https://github.com/nhabbash)
