# syllabus_diversity
This tool helps you analyze the diversity of assigned authors on course syllabi

It takes a list of assigned readings, and estimates the proportions of authors by gender and race/ethnicity.

For details of the methodology and results for urban sustainability courses, see:

Millard-Ball, Adam; Desai, Garima; and Fahrney, Jessica (2022). ["Diversifying Planning Education through Course Readings."](https://doi.org/10.1177%2F0739456X211001936) *Journal of Planning Education and Research*, in press.

Note that the estimates in the published paper use ethnicolr 0.4.0. This current version uses 0.8.1, with slightly different results.

You can also run the tool online at [www.syllabusdiversity.org](https://www.syllabusdiversity.org).

# How to install
Python 3.8 is required, along with the following Python packages. Other versions may also work, but have not been tested. 

```
numpy 1.19.2
pandas 1.3.5
ethnicolr 0.8.1
gender_guesser 0.4.0
Flask 2.0.3
h5py 3.1.0
Keras 2.2.4
tensorflow 2.5.2
```

Install them with:

`pip install numpy==1.19.2 pandas==1.3.5 ethnicolr==0.8.1 gender_guesser==0.4.0 Flask==2.0.3 h5py==3.1.0 Keras==2.2.4 tensorflow==2.5.2`

# Preparing your input file
Prepare your file with the list of authors or assigned readings using one of the two provided templates: `template1.csv` or `template2.csv`. Both .csv and Excel (.xlsx) formats are accepted, but for Excel, only the first sheet will be used.

In both cases, authors' names must be comma-separated, e.g. `Agyeman, Julian`. Authors without a comma will be ignored as they are assumed to be an institutional author (e.g. United Nations). 

## Option 1
`template1.csv` has one row for each reading. The (optional) `courseid` column enables you to compare results for different courses. It can be numeric or the name of a course. If you just have one course, you can omit this. The `reading` column is a citation in the following format: a list of authors (last name, first name) separated by semicolons, with ** after the last author. For example:

> Agyeman, Julian; Bullard, Robert; Evans, Bob** 2003. Just Sustainabilities: Development in an Unequal World. Cambridge, MA: MIT Press.

Everything after the ** is ignored

## Option 2
`template2.csv` has one row for each author. The `readingid` column is an identifier for each reading. This allows multi-authored publications to be weighted appropriately. The (optional) `courseid` column enables you to compare different courses. If you just have one course, you can omit this. The `author` column is the author's name in (lastname, firstname) format. For example:

> Agyeman, Julian
>
> Bullard, Robert

# Running the analysis
In the command line, type:
`python analyze_readings.py inputfilename.csv [outputfilename.csv]`

Summary statistics for each course will be displayed. If you include the optional `outputfilename.csv`, a new file will be created with the estimated gender and race/ethnicity for each author or reading.

Note that the estimates for race/ethnicity are probabilistic, i.e. the probability that an author falls into a given category. This works best with large numbers of authors.

# Questions?
We would love to hear about how this tool is being used. If you have any problems, please contact [Adam Millard-Ball](https://millardball.its.ucla.edu) or open a GitHub issue.
