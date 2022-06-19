---
title: Numerical analysis calculation work
author: Ivan Zhytkevych FI-91
language: ru-RU
header-includes: |
  \usepackage[utf8]{inputenc}
  \usepackage{fancyhdr}
  \pagestyle{fancy}
  \usepackage{hyperref}
  \usepackage{xcolor}
  \hypersetup{%
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=cyan,
    linkbordercolor={0 0 1}
  }
---

# Problem: variant №4
 
Варіант 4. 
Змоделювати процес розповсюдження газоподібної забруднюючої 
домішки в атмосфері за умови наявності ефекту самоочищення середовища.  
Самоочищення виникає при досягненні концентрацією $u$ деякого критичного 
значення.  
 
Рівняння еволюції концентрації домішки має вигляд: 

$$\frac{\partial u}{\partial t} = k_1 \frac{\partial^2 u}{\partial x^2} + k_2 \frac{\partial^2 u}{\partial y^2} + \mathcal{X} + \varphi$$

$$\mathcal{X} = \begin{cases} 0 &\quad u < u_{cr} \\ -du &\quad u \geq u_{cr} \end{cases}$$

$$u(0) = 0$$

$$u \mid_{\Gamma} = 0$$
 
Прийняти, що джерела забруднення розміщені в двох несусідніх 
центральних точках області, і діють з інтенсивністю $\varphi = 5\text{мкг/хв}$.
Розміри області завдати самостійно (але розбиття області повинно включати не менше 
10 вузлів за кожним напрямком). Знайти поле концентрації домішки на 
протязі часу, за який  концентрація домішки встигне досягти свого 
критичного значення хоча б в деяких точках області. 
