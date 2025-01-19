import math


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Line:
    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2

    def slope(self):
        if self.point2.x - self.point1.x == 0:
            return None
        return (self.point2.y - self.point1.y) / (self.point2.x - self.point1.x)

    def is_parallel(self, other_line):
        return self.slope() == other_line.slope()

    def is_perpendicular(self, other_line):
        slope1 = self.slope()
        slope2 = other_line.slope()
        if slope1 is None:
            return slope2 == 0
        if slope2 is None:
            return slope1 == 0
        return slope1 * slope2 == -1


class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def area(self):
        return math.pi * self.radius**2

    def intersects(self, other_circle):
        distance = math.sqrt(
            (self.center.x - other_circle.center.x) ** 2
            + (self.center.y - other_circle.center.y) ** 2
        )
        return distance <= self.radius + other_circle.radius


class Polygon:
    def __init__(self, vertices):
        self.vertices = vertices

    def perimeter(self):
        perimeter = 0
        for i in range(len(self.vertices)):
            point1 = self.vertices[i]
            point2 = self.vertices[(i + 1) % len(self.vertices)]
            perimeter += math.sqrt(
                (point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2
            )
        return perimeter


points = {
    "P1": Point(-6, 1),
    "P2": Point(2, 4),
    "P3": Point(-6, -1),
    "P4": Point(2, 2),
    "P5": Point(-4, -4),
    "P6": Point(-1, 6),
    "P7": Point(6, 3),
    "P8": Point(8, 1),
    "P9": Point(2, 0),
    "P10": Point(5, -1),
    "P11": Point(4, -4),
    "P12": Point(-1, -2),
}

line_a = Line(points["P1"], points["P2"])
line_b = Line(points["P3"], points["P4"])
line_c = Line(points["P5"], points["P6"])

circle_a = Circle(points["P7"], 2)
circle_b = Circle(points["P8"], 1)

polygon_a = Polygon([points["P9"], points["P10"], points["P11"], points["P12"]])


print("Are Line A and Line B parallel?", line_a.is_parallel(line_b))
print("Are Line C and Line A perpendicular?", line_c.is_perpendicular(line_a))
print("Area of Circle A:", circle_a.area())
print("Do Circle A and Circle B intersect?", circle_a.intersects(circle_b))
print("Perimeter of Polygon A:", polygon_a.perimeter())

print("-" * 40)


class Enemy:
    def __init__(self, label, x, y, vector, life_points=10):
        self.label = label
        self.x = x
        self.y = y
        self.vector = vector
        self.life_points = life_points
        self.alive = True

    def move(self):
        if self.alive:
            self.x += self.vector[0]
            self.y += self.vector[1]

    def take_damage(self, damage):
        if self.alive:
            self.life_points -= damage
            if self.life_points <= 0:
                self.life_points = 0
                self.alive = False

    def __str__(self):
        return f"{self.label}: Position=({self.x},{self.y}), Life={self.life_points}, Alive={self.alive}"


class Tower:
    def __init__(self, label, x, y, attack_points, attack_range):
        self.label = label
        self.x = x
        self.y = y
        self.attack_points = attack_points
        self.attack_range = attack_range

    def is_in_range(self, enemy):
        distance = math.sqrt((self.x - enemy.x) ** 2 + (self.y - enemy.y) ** 2)
        return distance <= self.attack_range

    def attack(self, enemies):
        for enemy in enemies:
            if self.is_in_range(enemy):
                enemy.take_damage(self.attack_points)


enemies = [
    Enemy("E1", -10, 2, (2, -1)),
    Enemy("E2", -8, 0, (3, 1)),
    Enemy("E3", -9, -1, (3, 0)),
]

towers = [
    Tower("T1", -3, 2, attack_points=1, attack_range=2),
    Tower("T2", -1, -2, attack_points=1, attack_range=2),
    Tower("T3", 4, 2, attack_points=1, attack_range=2),
    Tower("T4", 7, 0, attack_points=1, attack_range=2),
]

advanced_towers = [
    Tower("A1", 1, 1, attack_points=2, attack_range=4),
    Tower("A2", 4, -3, attack_points=2, attack_range=4),
]

for turn in range(10):
    for enemy in enemies:
        enemy.move()

    for tower in towers + advanced_towers:
        tower.attack(enemies)

print("Final Results:")
for enemy in enemies:
    print(enemy)
